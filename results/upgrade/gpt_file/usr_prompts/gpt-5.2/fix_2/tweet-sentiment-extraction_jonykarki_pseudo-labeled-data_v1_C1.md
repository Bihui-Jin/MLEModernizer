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

0.7075726985931396

# 6. Current score

0.59324

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.59324) has done: 'I remove the failing `fairseq/fastBPE` dependency (it isn’t available) and replace it with a compatible local Hugging Face tokenizer/model loader for the provided BERTweet files, keeping the same start/end-span inference logic. I also fix the Transformers API usage: load config/model from a local directory instead of passing a config.json path, and update the forward pass to use `hidden_states` correctly. To prevent DataLoader worker crashes due to missing globals, I set `num_workers=0` and ensure `bpe`/`vocab` equivalents are initialized before dataset creation. Finally, I guarantee the submission length matches `sample_submission.csv` and write `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch.nn as nn
import transformers
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

import warnings

warnings.filterwarnings("ignore")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
LEARNING_RATE = 6e-5
MAX_LEN = 126
TRAIN_BATCH_SIZE = 35
VALID_BATCH_SIZE = 32
EPOCHS = 3

INPUT_PATH = "/kaggle/input/"
MODEL_INPUT_PATH = f"{INPUT_PATH}bertweet-model/"
BERTWEET_DIR = f"{MODEL_INPUT_PATH}BERTweet_base_transformers"

tokenizer = RobertaTokenizerFast.from_pretrained(BERTWEET_DIR, local_files_only=True)


class _BPEWrap:
    def __init__(self, tok):
        self.tok = tok

    def encode(self, text: str) -> str:
        toks = self.tok.tokenize(text)
        return " ".join(toks)

    def decode(self, token_str: str) -> str:
        toks = token_str.split()
        return self.tok.convert_tokens_to_string(toks)


bpe = _BPEWrap(tokenizer)


class _VocabWrap:
    def __init__(self, tok):
        self.tok = tok
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 1

    def encode_line(self, token_str: str, append_eos=False, add_if_not_exist=False):
        toks = token_str.strip().split()
        ids = self.tok.convert_tokens_to_ids(toks)
        unk = self.tok.unk_token_id
        ids = [unk if (i is None or i < 0) else i for i in ids]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> str:
        return self.tok.convert_ids_to_tokens(int(idx))


vocab = _VocabWrap(tokenizer)




## --- ERROR in cell 1, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/bertweet-model/BERTweet_base_transformers'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/714245827.py in <cell line: 0>()
     12 # ---- Bugfix: fairseq/fastBPE isn't available. Use the local HF tokenizer that ships with the BERTweet folder.
     13 # This preserves the core idea: tokenize tweet + tokenize sentiment and predict start/end token indices.
---> 14 tokenizer = RobertaTokenizerFast.from_pretrained(BERTWEET_DIR, local_files_only=True)
     15 
     16 

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   1930                     except Exception:
   1931                         # For any other exception, we throw a generic error.
-> 1932                         raise OSError(
   1933                             f"Can't load tokenizer for '{pretrained_model_name_or_path}'. If you were trying to load it from "
   1934                             "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "

OSError: Can't load tokenizer for '/kaggle/input/bertweet-model/BERTweet_base_transformers'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/kaggle/input/bertweet-model/BERTweet_base_transformers' is the correct path to a directory containing all relevant files for a RobertaTokenizerFast tokenizer.

## === cell 2
class TweetDataset:
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
        e_tweet = "<s> " + bpe.encode(self.tweets[item]) + " </s>"
        enc_tweet = (
            vocab.encode_line(e_tweet, append_eos=False, add_if_not_exist=False)
            .long()
            .tolist()
        )

        if self.sentiments[item].strip() != "neutral":
            e_sentiment = (
                "</s> "
                + bpe.encode(self.sentiments[item])
                + " "
                + bpe.encode(self.sentiments[item])
                + " </s>"
            )
        else:
            e_sentiment = "</s> " + bpe.encode(self.sentiments[item]) + " </s>"
        enc_sentiment = (
            vocab.encode_line(e_sentiment, append_eos=False, add_if_not_exist=False)
            .long()
            .tolist()
        )

        enc_tweet_sentiment = enc_tweet + enc_sentiment

        if len(enc_tweet_sentiment) > self.max_len:
            enc_tweet_sentiment = enc_tweet_sentiment[: self.max_len]

        padding_len = self.max_len - len(enc_tweet_sentiment)
        pad_id = vocab.pad_id
        input_ids = enc_tweet_sentiment + ([pad_id] * padding_len)
        attention_mask = ([1] * len(enc_tweet_sentiment)) + ([0] * padding_len)

        start_index, end_index = 0, 0
        token_type_ids = [0] * self.max_len

        e_selected_text_ids = bpe.encode(self.selected_texts[item])
        enc_selected_text_ids = (
            vocab.encode_line(
                e_selected_text_ids, append_eos=False, add_if_not_exist=False
            )
            .long()
            .tolist()
        )

        if len(enc_selected_text_ids) > 0:
            for j in (
                i
                for i, e in enumerate(enc_tweet_sentiment)
                if e == enc_selected_text_ids[0]
            ):
                if (
                    enc_tweet_sentiment[j : j + len(enc_selected_text_ids)]
                    == enc_selected_text_ids
                ):
                    start_index = j
                    end_index = j + (len(enc_selected_text_ids))
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
        self.roberta = RobertaModel.from_pretrained(
            BERTWEET_DIR, config=conf, local_files_only=True
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
        hidden_states = out.hidden_states  # tuple: embeddings + layers

        hs_last = hidden_states[-1]
        hs_prev = hidden_states[-2]
        cat = torch.cat((hs_last, hs_prev), dim=-1)
        cat = self.drop_out(cat)
        logits = self.l0(cat)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 4
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values



## === cell 5
model_config = RobertaConfig.from_pretrained(BERTWEET_DIR, local_files_only=True)
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
/tmp/ipykernel_55/2511075590.py in <cell line: 0>()
      1 # ---- Bugfix: load config from local directory (not config.json path) to avoid HFValidationError.
----> 2 model_config = RobertaConfig.from_pretrained(BERTWEET_DIR, local_files_only=True)
      3 model_config.output_hidden_states = True
      4 

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
    test_dataset, shuffle=False, batch_size=VALID_BATCH_SIZE, num_workers=0
)



## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(weight_path: str):
    m = TweetModel(conf=model_config).to(device)
    state = torch.load(weight_path, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m


model1 = _load_model(f"{INPUT_PATH}twitroberta/model_0.bin")
model2 = _load_model(f"{INPUT_PATH}twitroberta/model_1.bin")
model3 = _load_model(f"{INPUT_PATH}twitroberta/model_2.bin")
model4 = _load_model(f"{INPUT_PATH}twitroberta/model_3.bin")
model5 = _load_model(f"{INPUT_PATH}twitroberta/model_4.bin")
model6 = _load_model(f"{INPUT_PATH}twitroberta/model_5.bin")
model7 = _load_model(f"{INPUT_PATH}twitroberta/model_6.bin")
model8 = _load_model(f"{INPUT_PATH}twitroberta/model_7.bin")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/998304208.py in <cell line: 0>()
     11 
     12 
---> 13 model1 = _load_model(f"{INPUT_PATH}twitroberta/model_0.bin")
     14 model2 = _load_model(f"{INPUT_PATH}twitroberta/model_1.bin")
     15 model3 = _load_model(f"{INPUT_PATH}twitroberta/model_2.bin")

/tmp/ipykernel_55/998304208.py in _load_model(weight_path)
      4 # Load ensemble weights (paths unchanged)
      5 def _load_model(weight_path: str):
----> 6     m = TweetModel(conf=model_config).to(device)
      7     state = torch.load(weight_path, map_location="cpu")
      8     m.load_state_dict(state)

NameError: name 'model_config' is not defined

## === cell 8
final_output = []
with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)

        orig_tweet = d["orig_tweet"]

        outputs_start1, outputs_end1 = model1(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start2, outputs_end2 = model2(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start3, outputs_end3 = model3(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start4, outputs_end4 = model4(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start5, outputs_end5 = model5(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start6, outputs_end6 = model6(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start7, outputs_end7 = model7(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )
        outputs_start8, outputs_end8 = model8(
            ids=ids, mask=mask, token_type_ids=token_type_ids
        )

        outputs_start = (
            outputs_start1
            + outputs_start2
            + outputs_start3
            + outputs_start4
            + outputs_start5
            + outputs_start7
            + outputs_start6
            + outputs_start8
        ) / 8

        outputs_end = (
            outputs_end1
            + outputs_end2
            + outputs_end3
            + outputs_end4
            + outputs_end5
            + outputs_end6
            + outputs_end7
            + outputs_end8
        ) / 8

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().numpy()

        for i, tweet in enumerate(orig_tweet):
            a = int(np.argmax(outputs_start[i]))
            b = int(np.argmax(outputs_end[i]))

            if a > b:
                selected_text = tweet
            else:
                n_tweet = "<s> " + bpe.encode(tweet) + " </s>"
                nn_tweet = (
                    vocab.encode_line(n_tweet, append_eos=False, add_if_not_exist=False)
                    .long()
                    .tolist()
                )
                nn_tweet = nn_tweet[:MAX_LEN]
                select_ids = nn_tweet[a:b]
                if len(select_ids) == 0:
                    selected_text = tweet
                else:
                    selected_text = bpe.decode(" ".join([vocab[t] for t in select_ids]))
                    selected_text = selected_text.replace("<s>", "").replace("</s>", "")
                    if selected_text.strip() == "":
                        selected_text = tweet

            final_output.append(selected_text)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/638843268.py in <cell line: 0>()
      2 with torch.no_grad():
      3     tk0 = tqdm(data_loader, total=len(data_loader))
----> 4     for bi, d in enumerate(tk0):
      5         ids = d["ids"].to(device, dtype=torch.long)
      6         token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)

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

/tmp/ipykernel_55/3351513678.py in __getitem__(self, item)
     15 
     16     def __getitem__(self, item):
---> 17         e_tweet = "<s> " + bpe.encode(self.tweets[item]) + " </s>"
     18         enc_tweet = (
     19             vocab.encode_line(e_tweet, append_eos=False, add_if_not_exist=False)

NameError: name 'bpe' is not defined

## === cell 9
sub = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

if len(final_output) != len(sub):
    final_output = (final_output[: len(sub)] + df_test.text.tolist())[: len(sub)]

sub.loc[:, "selected_text"] = final_output
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)



## === cell 10
sub.head()
