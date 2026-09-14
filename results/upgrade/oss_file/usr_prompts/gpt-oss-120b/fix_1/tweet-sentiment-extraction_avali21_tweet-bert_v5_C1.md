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

No external packages required in the script and installed.

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

0.4356454908847809

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import torch
from torch.autograd import Variable
import copy
import math
import transformers
from transformers import AdamW

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
sample_submission = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

## === cell 2
train.dropna(inplace=True)

## === cell 3
def clean_text(text):
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

## === cell 4
train['text'] = train['text'].apply(lambda x:clean_text(x))
train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))

test['text'] = test['text'].apply(lambda x:clean_text(x))

## === cell 5
train_positive = train.loc[(train.sentiment == 'positive')]
train_neutral = train.loc[(train.sentiment == 'neutral')]
train_negative = train.loc[(train.sentiment == 'negative')]

test_positive = test.loc[(test.sentiment == 'positive')]
test_neutral = test.loc[(test.sentiment == 'neutral')]
test_negative = test.loc[(test.sentiment == 'negative')]

## === cell 6
train_all = [train_positive, train_neutral, train_negative]
test_all = [test_positive, test_neutral, test_negative]

## === cell 7
len(train_positive)

## === cell 8
len(train_all)

## === cell 9
test_positive.head()

## === cell 10
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    if (len(a) + len(b) - len(c)) == 0:
        return 1
    return float(len(c)) / (len(a) + len(b) - len(c))

## === cell 11
def from_predicted_positon_to_text(predicted, threshold, padded_tokens, tokens_dict, raw_text):   
    predicted[predicted >= threshold] = 1
    predicted[predicted < threshold] = 0


    index_matrix = []
    for i in range(len(predicted)):
        index = [i for i,p in enumerate(predicted[i, :].tolist()) if p == 1]
        index_matrix.append(index)

    first_last_word_index = []
    for i, j in zip(padded_tokens.tolist(), index_matrix):
        last = len(j)-1
        if last < 0:
            f_l = [-1, -1]
        else:
            while i[j[last]] == 0:
                last -= 1
                if  i[j[last]] == i[j[0]]:
                    break
            f_l = [i[j[0]], i[j[last]]]
        first_last_word_index.append(f_l)
    first_word = []
    last_word = []
    for f_l in first_last_word_index:
        f_word =  [k for k, v in tokens_dict.items() if v == f_l[0]]    
        l_word =  [k for k, v in tokens_dict.items() if v == f_l[1]]

        first_word.append(f_word)    
        last_word.append(l_word)



    predicted_val_text = []
    for f, l, text in zip(first_word, last_word, raw_text):
        if len(f) == 0 or len(l) == 0:
            p_text = text
        else:
            f_index = text.index(f[0])
            l_index = text.index(l[0]) + len(l[0])
            p_text = text[f_index:l_index]
        predicted_val_text.append(p_text)
    return predicted_val_text

## === cell 12
class BertModel(torch.nn.Module):
    def __init__(self,UNCASED, outputSize, droupout, std):
        super(BertModel, self).__init__()
        self.config = transformers.BertConfig.from_pretrained(UNCASED, output_hidden_states=True)
        self.bert_model = transformers.BertModel.from_pretrained(UNCASED, config=self.config)
        self.drop_out = torch.nn.Dropout(droupout)
        self.linear_model = torch.nn.Linear(768, outputSize)
        torch.nn.init.normal_(self.linear_model.weight, std=std)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
            
        _,_,hidden_states = self.bert_model(input_ids, attention_mask=attention_mask.float())
        embedding_output = hidden_states[0]
        attention_hidden_states = hidden_states[1:]

        summed_last_4_layers = torch.stack(attention_hidden_states[-4:]).sum(0)
        sentence_embedding = torch.mean(summed_last_4_layers, dim=1)

        out = self.drop_out(sentence_embedding)
        out = self.linear_model(out)
        out = self.sigmoid(out)
        return out

## === cell 13
torch.cuda.empty_cache()
category = ["positive", "neutral", "negative"]
color = ['b', 'r', 'g']
max_sequence_length = 32
val_frac = 0.25
num_of_val = 400
learningRate = 3e-4 # 0.01
threshold = 0.80
max_length = max_sequence_length
outputSize = max_sequence_length
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = [10, 3, 9]
predicted_text = {}

VOCAB='/kaggle/input/bertbaseuncased/vocab.txt' # your path for model and vocab 
UNCASED='/kaggle/input/bertbaseuncased/'
std = 0.02
droupout = 0.1
batch_size = num_of_val
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
tokens_dict = {}


for i in range(len(train_all)):

    
    tokenizer = transformers.BertTokenizer.from_pretrained(VOCAB)
    
    for k, v in list(tokenizer.vocab.items()):
        if k[1:-1] not in ["SEP", "MASK", "CLS"]:
            tokens_dict[k[1:-1]] = v
            
    model = BertModel(UNCASED, outputSize, droupout, std)
    model.to(device)
    criterion = torch.nn.MSELoss()
    param_optimizer = list(model.named_parameters())
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=learningRate)
    
    train_data = train_all[i]
    test_data = test_all[i]
    
    
    X_train_text = train_data.loc[:, "text"].tolist()
    y_train_text = train_data.loc[:, "selected_text"].tolist()
    X_test = test_data.loc[:, ["textID", "text"]]
    X_train_tokens = [tokenizer.encode(t, add_special_tokens=True, max_length = max_length) for t in X_train_text]    
    y_train_tokens = [tokenizer.encode(t, add_special_tokens=True, max_length = max_length) for t in y_train_text]
    X_test_tokens = [tokenizer.encode(t, add_special_tokens=True, max_length = max_length) for t in X_test.text]

    X_train_all_input_ids = np.array([i + [0]*(max_length - len(i)) for i in X_train_tokens])
    y_train_all_input_ids = np.array([i + [0]*(max_length - len(i)) for i in y_train_tokens])
    X_test_input_ids = np.array([i + [0]*(max_length - len(i)) for i in X_test_tokens])
    X_train_all_attention_mask = np.where(X_train_all_input_ids != 0, 1, 0)
    y_train_all_attention_mask = np.where(y_train_all_input_ids != 0, 1, 0)
    X_test_attention_mask = np.where(X_test_input_ids != 0, 1, 0)
    
    y_train_bool = []
    for j in range(len(y_train_all_input_ids)):
        a = [1 if x > 0 and x in y_train_all_input_ids[j,:] else 0 for x in X_train_all_input_ids[j,:].tolist()]
        y_train_bool.append(a)
    y_train_bool = np.array(y_train_bool)
    
    
    

    num_of_val = num_of_val + (len(X_train_all_input_ids) - num_of_val)%batch_size
    num_of_train = len(X_train_all_input_ids) - num_of_val



    X_val = X_train_all_input_ids[-num_of_val:,:]
    X_val_attention_mask = X_train_all_attention_mask[-num_of_val:,:]
    y_val = y_train_bool[-num_of_val:,:]
    
    optimizer.zero_grad()
    training_loss = []
    
        
        
    for epoch in range(epochs[i]):
        
        for k_fold in range(int(num_of_train/batch_size)):
            X_train = X_train_all_input_ids[k_fold*batch_size:(k_fold+1)*batch_size,:]
            X_train_attention_mask = X_train_all_attention_mask[k_fold*batch_size:(k_fold+1)*batch_size,:]
            y_train = y_train_bool[k_fold*batch_size:(k_fold+1)*batch_size,:]
            
            X_train =  Variable(torch.from_numpy(X_train).to(device))
            X_train_attention_mask = Variable(torch.from_numpy(X_train_attention_mask).to(device))
            y_train = Variable(torch.from_numpy(y_train).to(device))

            outputs = model(X_train, X_train_attention_mask)
            loss = criterion(outputs.float(), y_train.float())
            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
            if k_fold%1 == 0:
                print('{}-th category, total {} fold, now {}, epoch {}, loss {}'.format(i, int(num_of_train/batch_size), k_fold, epoch, loss.item()))

    
    plt.plot(range(len(training_loss)), training_loss, color[i], label='training loss')
    
    
    with torch.no_grad(): # we don't need gradients in the testing phase

        X_val =  Variable(torch.from_numpy(X_val).to(device))
        X_val_attention_mask =  Variable(torch.from_numpy(X_val_attention_mask).to(device))
        y_val =  Variable(torch.from_numpy(y_val).to(device))

        eval_predicted = model(X_val, X_val_attention_mask)
    
    predicted_val_text = from_predicted_positon_to_text(eval_predicted, threshold, X_val, tokens_dict,  X_train_text[-num_of_val:]) 
    jaccard_score_list = []
    for str1, str2 in zip(predicted_val_text, y_train_text[-num_of_val:]):
        jaccard_score_list.append(jaccard(str1, str2))
    result = pd.Series(jaccard_score_list)
    print(result.describe())
    
    
    with torch.no_grad(): # we don't need gradients in the testing phase
        X_test_input_ids =  Variable(torch.from_numpy(X_test_input_ids).to(device))
        X_test_attention_mask =  Variable(torch.from_numpy(X_test_attention_mask).to(device))
        
        test_predicted = model(X_test_input_ids, X_test_attention_mask)
        
    predicted_test_text = from_predicted_positon_to_text(test_predicted, threshold, X_test_input_ids, tokens_dict, X_test.text) 
    for p, idx in zip(predicted_test_text, X_test.textID.tolist()):
        predicted_text[idx] = p

## --- ERROR in cell 13, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/bertbaseuncased/vocab.txt'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   1911                     try:
-> 1912                         resolved_config_file = cached_file(
   1913                             pretrained_model_name_or_path,

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/bertbaseuncased/vocab.txt'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2207933502.py in <cell line: 0>()
     30 
     31     ##################################################### Model Parameters #####################################################
---> 32     tokenizer = transformers.BertTokenizer.from_pretrained(VOCAB)
     33 
     34     for k, v in list(tokenizer.vocab.items()):

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   1930                     except Exception:
   1931                         # For any other exception, we throw a generic error.
-> 1932                         raise OSError(
   1933                             f"Can't load tokenizer for '{pretrained_model_name_or_path}'. If you were trying to load it from "
   1934                             "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "

OSError: Can't load tokenizer for '/kaggle/input/bertbaseuncased/vocab.txt'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/kaggle/input/bertbaseuncased/vocab.txt' is the correct path to a directory containing all relevant files for a BertTokenizer tokenizer.

## === cell 14
sample_submission_df = pd.DataFrame()
for idx, row in sample_submission.iterrows():
    row["selected_text"] = predicted_text[row["textID"]]
    row_frame = row.to_frame().T
    if sample_submission_df.empty:
        sample_submission_df = row_frame
    else:
        sample_submission_df = sample_submission_df.append(row_frame)
display(sample_submission_df)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/551555338.py in <cell line: 0>()
      1 sample_submission_df = pd.DataFrame()
      2 for idx, row in sample_submission.iterrows():
----> 3     row["selected_text"] = predicted_text[row["textID"]]
      4     row_frame = row.to_frame().T
      5     if sample_submission_df.empty:

KeyError: '80a1e6bc32'

## === cell 15
sample_submission_df.to_csv("submission.csv", index = False)

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame must have a 'textID' column.
