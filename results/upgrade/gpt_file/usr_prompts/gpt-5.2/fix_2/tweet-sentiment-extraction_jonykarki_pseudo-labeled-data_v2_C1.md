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

0.6645270586013794

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I remove the unavailable `fairseq/fastBPE` dependency (it cannot be imported in this environment) and replace it with the locally-available `tokenizers` ByteLevelBPETokenizer built from the provided BERTweet `bpe.codes`/`dict.txt`, keeping the same overall inference approach. I also fix the Hugging Face config/model loading to use local paths correctly with this transformers version, and make the Roberta forward call compatible (use `output_hidden_states=True` and read `hidden_states`). Finally, I prevent DataLoader worker crashes by using `num_workers=0`, ensure deterministic inference (remove random span jitter), clamp span indices safely, and write a submission CSV whose row count exactly matches `test.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn

import transformers
from tokenizers import ByteLevelBPETokenizer

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
LEARNING_RATE = 6e-5
MAX_LEN = 126
TRAIN_BATCH_SIZE = 35
VALID_BATCH_SIZE = 32
EPOCHS = 3

INPUT_PATH = "/kaggle/input/"
MODEL_INPUT_PATH = f"{INPUT_PATH}bertweet-model/"
BERTWEET_DIR = f"{MODEL_INPUT_PATH}BERTweet_base_transformers"

TEST_CSV = "/kaggle/input/tweet-sentiment-extraction/test.csv"
SAMPLE_SUB = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

dict_path = Path(BERTWEET_DIR) / "dict.txt"
codes_path = Path(BERTWEET_DIR) / "bpe.codes"
assert dict_path.exists(), f"Missing {dict_path}"
assert codes_path.exists(), f"Missing {codes_path}"

tmp_vocab_json = Path("/kaggle/working/bertweet_vocab.json")

token_to_id = {}
with open(dict_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split()
        if not parts:
            continue
        tok = parts[0]
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)

with open(tmp_vocab_json, "w", encoding="utf-8") as f:
    json.dump(token_to_id, f, ensure_ascii=False)

bpe_tokenizer = ByteLevelBPETokenizer(
    vocab=str(tmp_vocab_json),
    merges=str(codes_path),
    add_prefix_space=True,
)

id_to_token = {i: t for t, i in token_to_id.items()}
PAD_ID = token_to_id.get("<pad>", 1)
BOS_ID = token_to_id.get("<s>", 0)
EOS_ID = token_to_id.get("</s>", 2)


def bpe_encode_to_ids(text: str):
    text = " " + " ".join(str(text).split())
    enc = bpe_tokenizer.encode(text)
    return enc.ids


def bpe_decode_from_ids(ids):
    s = bpe_tokenizer.decode(ids)
    s = s.replace("<s>", "").replace("</s>", "")
    return s




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/967688654.py in <cell line: 0>()
     16 dict_path = Path(BERTWEET_DIR) / "dict.txt"
     17 codes_path = Path(BERTWEET_DIR) / "bpe.codes"
---> 18 assert dict_path.exists(), f"Missing {dict_path}"
     19 assert codes_path.exists(), f"Missing {codes_path}"
     20 

AssertionError: Missing /kaggle/input/bertweet-model/BERTweet_base_transformers/dict.txt

## === cell 2
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = [" " + " ".join(str(tweet).split()) for tweet in tweets]
        self.sentiments = [
            " " + " ".join(str(sentiment).split()) for sentiment in sentiments
        ]
        self.selected_texts = [
            " " + " ".join(str(selected_text).split())
            for selected_text in selected_texts
        ]
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet_ids = [BOS_ID] + bpe_encode_to_ids(self.tweets[item]) + [EOS_ID]

        sent = self.sentiments[item].strip()
        sent_ids_core = bpe_encode_to_ids(sent)
        if sent != "neutral":
            sentiment_ids = [EOS_ID] + sent_ids_core + sent_ids_core + [EOS_ID]
        else:
            sentiment_ids = [EOS_ID] + sent_ids_core + [EOS_ID]

        enc_tweet_sentiment = tweet_ids + sentiment_ids

        enc_tweet_sentiment = enc_tweet_sentiment[: self.max_len]
        attention_mask = [1] * len(enc_tweet_sentiment)
        padding_len = self.max_len - len(enc_tweet_sentiment)

        input_ids = enc_tweet_sentiment + ([PAD_ID] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        token_type_ids = [0] * self.max_len

        start_index, end_index = 0, 0
        sel_ids = bpe_encode_to_ids(self.selected_texts[item])
        if len(sel_ids) > 0:
            for j in (i for i, e in enumerate(enc_tweet_sentiment) if e == sel_ids[0]):
                if enc_tweet_sentiment[j : j + len(sel_ids)] == sel_ids:
                    start_index = j
                    end_index = j + len(sel_ids)
                    break

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets_start": torch.tensor(start_index, dtype=torch.long),
            "targets_end": torch.tensor(end_index, dtype=torch.long),
            "orig_tweet": self.tweets[item],
            "orig_selected": self.selected_texts[item],
            "sentiment": self.sentiments[item],
        }




## === cell 3
class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            BERTWEET_DIR,
            config=conf,
            local_files_only=True,
        )
        self.drop_out = nn.Dropout(0.1)
        self.activation = nn.LeakyReLU()
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        out = self.roberta(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states  # tuple(layer0..layerN)
        hs = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        hs = self.drop_out(hs)
        logits = self.l0(hs)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
df_test = pd.read_csv(TEST_CSV)
df_test.loc[:, "selected_text"] = df_test["text"].values



## === cell 5
model_config = transformers.RobertaConfig.from_pretrained(
    BERTWEET_DIR,
    local_files_only=True,
)
model_config.output_hidden_states = True



## --- ERROR in cell 5, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/bertweet-model/BERTweet_base_transformers'. Use `repo_type` argument if needed.

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/bertweet-model/BERTweet_base_transformers'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/3260150065.py in <cell line: 0>()
      1 # --- Bugfix: from_pretrained should point to a directory (local), not a config.json file path.
----> 2 model_config = transformers.RobertaConfig.from_pretrained(
      3     BERTWEET_DIR,
      4     local_files_only=True,
      5 )

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

OSError: Can't load the configuration of '/kaggle/input/bertweet-model/BERTweet_base_transformers'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/kaggle/input/bertweet-model/BERTweet_base_transformers' is the correct path to a directory containing a config.json file

## === cell 6
test_dataset = TweetDataset(
    tweets=df_test.text.values,
    sentiments=df_test.sentiment.values,
    selected_texts=df_test.selected_text.values,
)

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=VALID_BATCH_SIZE,
    num_workers=0,
)



## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = [f"{INPUT_PATH}twitroberta/model_{i}.bin" for i in range(8)]
models = []
for mp in model_paths:
    m = TweetModel(conf=model_config).to(device)
    state = torch.load(mp, map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()
    models.append(m)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3622501196.py in <cell line: 0>()
      5 models = []
      6 for mp in model_paths:
----> 7     m = TweetModel(conf=model_config).to(device)
      8     state = torch.load(mp, map_location="cpu")
      9     m.load_state_dict(state, strict=True)

NameError: name 'model_config' is not defined

## === cell 8
final_output = []

with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for _, d in enumerate(tk0):
        ids = d["ids"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)

        orig_tweet = d["orig_tweet"]

        outs_start = []
        outs_end = []
        for m in models:
            s, e = m(ids=ids, mask=mask, token_type_ids=token_type_ids)
            outs_start.append(s)
            outs_end.append(e)

        outputs_start = torch.stack(outs_start, dim=0).mean(dim=0)
        outputs_end = torch.stack(outs_end, dim=0).mean(dim=0)

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().numpy()

        for i, tweet in enumerate(orig_tweet):
            a = int(np.argmax(outputs_start[i]))
            b = int(np.argmax(outputs_end[i]))

            a = max(0, min(a, MAX_LEN - 1))
            b = max(0, min(b, MAX_LEN))

            if a > b:
                selected_text = tweet
            else:
                tweet_ids = [BOS_ID] + bpe_encode_to_ids(tweet) + [EOS_ID]
                select_ids = tweet_ids[a:b] if b > a else []
                selected_text = bpe_decode_from_ids(select_ids).strip()
                if selected_text == "":
                    selected_text = tweet

            final_output.append(selected_text)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1478988061.py in <cell line: 0>()
      3 with torch.no_grad():
      4     tk0 = tqdm(data_loader, total=len(data_loader))
----> 5     for _, d in enumerate(tk0):
      6         ids = d["ids"].to(device, dtype=torch.long)
      7         token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)

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
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/2755884033.py in __getitem__(self, item)
     15 
     16     def __getitem__(self, item):
---> 17         tweet_ids = [BOS_ID] + bpe_encode_to_ids(self.tweets[item]) + [EOS_ID]
     18 
     19         sent = self.sentiments[item].strip()

NameError: name 'BOS_ID' is not defined

## === cell 9
sub = pd.read_csv(SAMPLE_SUB)
if len(final_output) != len(sub):
    final_output = final_output[: len(sub)]
    if len(final_output) < len(sub):
        final_output = final_output + [""] * (len(sub) - len(final_output))

sub.loc[:, "selected_text"] = final_output
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)



## === cell 10
sub.sample(20, random_state=SEED)
