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
joblib==1.5.2
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

0.6745771765708923

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install /kaggle/input/fairseq-and-fastbpe/sacrebleu-1.4.9-py3-none-any.whl
!pip install /kaggle/input/fairseq-and-fastbpe/fairseq-0.9.0-cp37-cp37m-linux_x86_64.whl
!pip install /kaggle/input/fairseq-and-fastbpe/fastBPE-0.1.0-cp37-cp37m-linux_x86_64.whl

## === cell 1
import os
import torch
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers

import argparse
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1813362490.py in <cell line: 0>()
     15 
     16 import argparse
---> 17 from fairseq.data.encoders.fastbpe import fastBPE
     18 from fairseq.data import Dictionary
     19 

ModuleNotFoundError: No module named 'fairseq'

## === cell 2
LEARNING_RATE = 6e-5
MAX_LEN = 126
TRAIN_BATCH_SIZE = 35
VALID_BATCH_SIZE = 32
EPOCHS = 3
INPUT_PATH = "/kaggle/input/"
MODEL_INPUT_PATH = f"{INPUT_PATH}bertweet-model/"

key = argparse.Namespace(bpe_codes= f"{MODEL_INPUT_PATH}BERTweet_base_transformers/bpe.codes")
bpe = fastBPE(key)

vocab = Dictionary()
vocab.add_from_file(f"{MODEL_INPUT_PATH}BERTweet_base_transformers/dict.txt")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1053812333.py in <cell line: 0>()
      8 
      9 key = argparse.Namespace(bpe_codes= f"{MODEL_INPUT_PATH}BERTweet_base_transformers/bpe.codes")
---> 10 bpe = fastBPE(key)
     11 
     12 # Load the dictionary

NameError: name 'fastBPE' is not defined

## === cell 3
class TweetDataset:
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = [' '+' '.join(str(tweet).split()) for tweet in tweets]
        self.sentiments = [' '+' '.join(str(sentiment).split()) for sentiment in sentiments]
        self.selected_texts = [' '+' '.join(str(selected_text).split()) for selected_text in selected_texts]
        self.max_len = MAX_LEN
        
    
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        e_tweet = '<s> ' + bpe.encode(self.tweets[item]) + ' </s>'
        enc_tweet = vocab.encode_line(e_tweet, append_eos=False, add_if_not_exist=False).long().tolist()

        if self.sentiments[item] != 'neutral':
            e_sentiment = '</s> ' + bpe.encode(self.sentiments[item]) + " " + bpe.encode(self.sentiments[item]) + ' </s>'
        else:
            e_sentiment = '</s> ' + bpe.encode(self.sentiments[item]) + ' </s>'
        enc_sentiment = vocab.encode_line(e_sentiment, append_eos=False, add_if_not_exist=False).long().tolist()

        enc_tweet_sentiment = enc_tweet + enc_sentiment

        padding_len = self.max_len - len(enc_tweet_sentiment)
        input_ids = enc_tweet_sentiment + ([0] * padding_len)
        attention_mask = ([1] * len(enc_tweet_sentiment)) + ([0] * padding_len)

        start_index, end_index = 0, 0
        token_type_ids = [0] * self.max_len

        e_selected_text_ids = bpe.encode(self.selected_texts[item])
        enc_selected_text_ids = vocab.encode_line(e_selected_text_ids, append_eos=False, add_if_not_exist=False).long().tolist()

        for j in (i for i,e in enumerate(enc_tweet_sentiment) if e == enc_selected_text_ids[0]):
            if enc_tweet_sentiment[j:j+len(enc_selected_text_ids)] == enc_selected_text_ids:
                start_index = j
                end_index = j+(len(enc_selected_text_ids))

        return {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets_start': torch.tensor(start_index, dtype=torch.long),
            'targets_end': torch.tensor(end_index, dtype=torch.long),
            'orig_tweet': self.tweets[item],
            'orig_selected': self.selected_texts[item],
            'sentiment': self.sentiments[item]
        }

## === cell 4
class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            "/kaggle/input/bertweet-model/BERTweet_base_transformers/model.bin",
            config=conf
        )
        self.drop_out = nn.Dropout(0.1)
        self.activation = nn.LeakyReLU()
        self.l0 = nn.Linear(768 * 2, 2)

        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values

## === cell 6
model_config = transformers.RobertaConfig.from_pretrained(
                f"{MODEL_INPUT_PATH}BERTweet_base_transformers/config.json"
            )
model_config.output_hidden_states = True

## --- ERROR in cell 6, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/bertweet-model/BERTweet_base_transformers/config.json'. Use `repo_type` argument if needed.

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/bertweet-model/BERTweet_base_transformers/config.json'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/1577667582.py in <cell line: 0>()
----> 1 model_config = transformers.RobertaConfig.from_pretrained(
      2                 f"{MODEL_INPUT_PATH}BERTweet_base_transformers/config.json"
      3             )
      4 model_config.output_hidden_states = True

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

OSError: Can't load the configuration of '/kaggle/input/bertweet-model/BERTweet_base_transformers/config.json'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/kaggle/input/bertweet-model/BERTweet_base_transformers/config.json' is the correct path to a directory containing a config.json file

## === cell 7
test_dataset = TweetDataset(
        tweets=df_test.text.values,
        sentiments=df_test.sentiment.values,
        selected_texts=df_test.selected_text.values
    )

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=VALID_BATCH_SIZE,
    num_workers=1
)

## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = TweetModel(conf=model_config)
model1.to(device)
model1.load_state_dict(torch.load(f"{INPUT_PATH}twitroberta/model_0.bin"))
model1.eval()

model2 = TweetModel(conf=model_config)
model2.to(device)
model2.load_state_dict(torch.load(f"{INPUT_PATH}twitroberta/model_1.bin"))
model2.eval()

model3 = TweetModel(conf=model_config)
model3.to(device)
model3.load_state_dict(torch.load(f"{INPUT_PATH}twitroberta/model_2.bin"))
model3.eval()

model4 = TweetModel(conf=model_config)
model4.to(device)
model4.load_state_dict(torch.load(f"{INPUT_PATH}twitroberta/model_3.bin"))
model4.eval()

model5 = TweetModel(conf=model_config)
model5.to(device)
model5.load_state_dict(torch.load(f"{INPUT_PATH}twitroberta/model_4.bin"))
model5.eval()

model6 = TweetModel(conf=model_config)
model6.to(device)
model6.load_state_dict(torch.load(f"{INPUT_PATH}twitroberta/model_5.bin"))
model6.eval()

model7 = TweetModel(conf=model_config)
model7.to(device)
model7.load_state_dict(torch.load(f"{INPUT_PATH}twitroberta/model_6.bin"))
model7.eval()

model8 = TweetModel(conf=model_config)
model8.to(device)
model8.load_state_dict(torch.load(f"{INPUT_PATH}twitroberta/model_7.bin"))
model8.eval()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1922868218.py in <cell line: 0>()
      1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      2 
----> 3 model1 = TweetModel(conf=model_config)
      4 model1.to(device)
      5 model1.load_state_dict(torch.load(f"{INPUT_PATH}twitroberta/model_0.bin"))

NameError: name 'model_config' is not defined

## === cell 9
final_output = []
import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        outputs_start1, outputs_end1 = model1(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start2, outputs_end2 = model2(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start3, outputs_end3 = model3(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start4, outputs_end4 = model4(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start5, outputs_end5 = model5(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start6, outputs_end6 = model6(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start7, outputs_end7 = model7(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start8, outputs_end8 = model8(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start = (outputs_start1 + outputs_start2 + 
                         outputs_start3 + outputs_start4 + 
                         outputs_start5 + outputs_start7 + 
                         outputs_start6 + outputs_start8) / 8

        outputs_end = (outputs_end1 + outputs_end2 + outputs_end3 + outputs_end4 +
                       outputs_end5 + outputs_end6 + 
                       outputs_end7 + outputs_end8) / 8
        
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        
        for id, tweet in enumerate(orig_tweet):
            a = np.argmax(outputs_start[id])
            b = np.argmax(outputs_end[id])
            if a > b:
                selected_text = tweet
            else:
                n_tweet = '<s> ' + bpe.encode(tweet) + ' </s>'
                nn_tweet = vocab.encode_line(n_tweet, append_eos=False, add_if_not_exist=False).long().tolist()
                select = nn_tweet[a:b+1]
                selected_text = bpe.decode(" ".join([vocab[i] for i in select]))
                selected_text = selected_text.replace("<s>","").replace("</s>","").replace("<unk>", "")
                if selected_text.split()[-1] in set(string.punctuation) or selected_text.split()[-1] in stop_words:
                    selected_text = " ".join(selected_text.split()[:-1])
                if selected_text.strip() == "":
                    selected_text = tweet
            final_output.append(selected_text)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3453354633.py in <cell line: 0>()
      5 with torch.no_grad():
      6     tk0 = tqdm(data_loader, total=len(data_loader))
----> 7     for bi, d in enumerate(tk0):
      8         ids = d["ids"]
      9         token_type_ids = d["token_type_ids"]

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

NameError: Caught NameError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_11/3839430413.py", line 13, in __getitem__
    e_tweet = '<s> ' + bpe.encode(self.tweets[item]) + ' </s>'
                       ^^^
NameError: name 'bpe' is not defined


## === cell 10
a = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
a.loc[:, 'selected_text'] = final_output
a.to_csv("submission.csv", index=False)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3228082221.py in <cell line: 0>()
      1 a = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
----> 2 a.loc[:, 'selected_text'] = final_output
      3 a.to_csv("submission.csv", index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __setitem__(self, key, value)
    909 
    910         iloc = self if self.name == "iloc" else self.obj.iloc
--> 911         iloc._setitem_with_indexer(indexer, value, self.name)
    912 
    913     def _validate_key(self, key, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _setitem_with_indexer(self, indexer, value, name)
   1940         if take_split_path:
   1941             # We have to operate column-wise
-> 1942             self._setitem_with_indexer_split_path(indexer, value, name)
   1943         else:
   1944             self._setitem_single_block(indexer, value, name)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _setitem_with_indexer_split_path(self, indexer, value, name)
   1996                     return self._setitem_with_indexer((pi, info_axis[0]), value[0])
   1997 
-> 1998                 raise ValueError(
   1999                     "Must have equal len keys and value "
   2000                     "when setting with an iterable"

ValueError: Must have equal len keys and value when setting with an iterable

## === cell 11
a.sample(20)
