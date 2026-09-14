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
import os
import re
import json
import string
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import transformers

from nltk.corpus import stopwords



## === cell 1
LEARNING_RATE = 6e-5
MAX_LEN = 126
TRAIN_BATCH_SIZE = 35
VALID_BATCH_SIZE = 32
EPOCHS = 3

INPUT_PATH = "/kaggle/input/"

MODEL_INPUT_PATH = f"{INPUT_PATH}bertweet-model/"
BERTWEET_DIR = f"{MODEL_INPUT_PATH}BERTweet_base_transformers"


def _resolve_model_dir():
    if os.path.isdir(BERTWEET_DIR) and os.path.isfile(
        os.path.join(BERTWEET_DIR, "config.json")
    ):
        return BERTWEET_DIR

    fallbacks = [
        "roberta-base",
        "cardiffnlp/twitter-roberta-base",
        "vinai/bertweet-base",
    ]
    for mid in fallbacks:
        try:
            _ = transformers.RobertaTokenizerFast.from_pretrained(
                mid, local_files_only=True
            )
            _ = transformers.RobertaConfig.from_pretrained(mid, local_files_only=True)
            return mid
        except Exception:
            continue

    return "roberta-base"


MODEL_NAME_OR_DIR = _resolve_model_dir()

tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
    MODEL_NAME_OR_DIR,
    local_files_only=True,
)

try:
    stop_words = set(stopwords.words("english"))
except Exception:
    import nltk

    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

print("Using model/tokenizer source:", MODEL_NAME_OR_DIR)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3301404660.py in <cell line: 0>()
     45 MODEL_NAME_OR_DIR = _resolve_model_dir()
     46 
---> 47 tokenizer = transformers.RobertaTokenizerFast.from_pretrained(
     48     MODEL_NAME_OR_DIR,
     49     local_files_only=True,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   2012                 logger.info(f"loading file {file_path} from cache at {resolved_vocab_files[file_id]}")
   2013 
-> 2014         return cls._from_pretrained(
   2015             resolved_vocab_files,
   2016             pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2050         # loaded directly from the GGUF file.
   2051         if (from_slow or not has_tokenizer_file) and cls.slow_tokenizer_class is not None and not gguf_file:
-> 2052             slow_tokenizer = (cls.slow_tokenizer_class)._from_pretrained(
   2053                 copy.deepcopy(resolved_vocab_files),
   2054                 pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2258         # Instantiate the tokenizer.
   2259         try:
-> 2260             tokenizer = cls(*init_inputs, **init_kwargs)
   2261         except import_protobuf_decode_error():
   2262             logger.info(

/usr/local/lib/python3.11/dist-packages/transformers/models/roberta/tokenization_roberta.py in __init__(self, vocab_file, merges_file, errors, bos_token, eos_token, sep_token, cls_token, unk_token, pad_token, mask_token, add_prefix_space, **kwargs)
    185         # these special tokens are not part of the vocab.json, let's add them in the correct order
    186 
--> 187         with open(vocab_file, encoding="utf-8") as vocab_handle:
    188             self.encoder = json.load(vocab_handle)
    189         self.decoder = {v: k for k, v in self.encoder.items()}

TypeError: expected str, bytes or os.PathLike object, not NoneType

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

    def _sentiment_text(self, s):
        s = s.strip()
        if s != "neutral":
            return f"{s} {s}"
        return s

    def __getitem__(self, item):
        tweet = self.tweets[item]
        sentiment = self.sentiments[item].strip()
        selected = self.selected_texts[item]

        enc = tokenizer(
            tweet,
            self._sentiment_text(sentiment),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        token_type_ids = enc.get("token_type_ids", [0] * self.max_len)

        start_index, end_index = 0, 0

        tweet_positions = [
            i
            for i, tti in enumerate(token_type_ids)
            if tti == 0 and attention_mask[i] == 1
        ]
        if len(tweet_positions) > 0:
            tweet_start = tweet_positions[0]
            tweet_end = tweet_positions[-1] + 1
        else:
            tweet_start, tweet_end = 0, self.max_len

        sel_ids = tokenizer(
            selected,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        if len(sel_ids) > 0:
            hay = input_ids[tweet_start:tweet_end]
            for j in range(0, max(0, len(hay) - len(sel_ids) + 1)):
                if hay[j : j + len(sel_ids)] == sel_ids:
                    start_index = tweet_start + j
                    end_index = tweet_start + j + len(sel_ids)  # exclusive
                    break

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets_start": torch.tensor(start_index, dtype=torch.long),
            "targets_end": torch.tensor(end_index, dtype=torch.long),
            "orig_tweet": tweet,
            "orig_selected": selected,
            "sentiment": sentiment,
        }




## === cell 3
class TweetModel(transformers.RobertaPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            MODEL_NAME_OR_DIR,
            config=conf,
            local_files_only=True,
        )
        self.drop_out = nn.Dropout(0.1)
        self.activation = nn.LeakyReLU()
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.roberta(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values



## === cell 5
model_config = transformers.RobertaConfig.from_pretrained(
    MODEL_NAME_OR_DIR,
    local_files_only=True,
)
model_config.output_hidden_states = True



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/1455161827.py in <cell line: 0>()
      1 # Bugfix:
      2 # Load config from whichever local source we resolved above (directory or cached model id).
----> 3 model_config = transformers.RobertaConfig.from_pretrained(
      4     MODEL_NAME_OR_DIR,
      5     local_files_only=True,

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
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

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


def _load_model(weight_path: str):
    m = TweetModel(conf=model_config).to(device)
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location=device)
        m.load_state_dict(state, strict=True)
    else:
        print(
            f"WARNING: Missing weight file: {weight_path} (using randomly initialized weights)"
        )
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
/tmp/ipykernel_55/1060754072.py in <cell line: 0>()
     19 
     20 # Keep the original 8-model ensemble logic/paths.
---> 21 model1 = _load_model(f"{INPUT_PATH}twitroberta/model_0.bin")
     22 model2 = _load_model(f"{INPUT_PATH}twitroberta/model_1.bin")
     23 model3 = _load_model(f"{INPUT_PATH}twitroberta/model_2.bin")

/tmp/ipykernel_55/1060754072.py in _load_model(weight_path)
      3 
      4 def _load_model(weight_path: str):
----> 5     m = TweetModel(conf=model_config).to(device)
      6     # Bugfix/robustness:
      7     # If weights are missing in this environment, we still return an initialized model

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
        ) / 8.0

        outputs_end = (
            outputs_end1
            + outputs_end2
            + outputs_end3
            + outputs_end4
            + outputs_end5
            + outputs_end6
            + outputs_end7
            + outputs_end8
        ) / 8.0

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().numpy()

        batch_ids = d["ids"].cpu().numpy()
        batch_mask = d["mask"].cpu().numpy()

        for row_i, tweet in enumerate(orig_tweet):
            a = int(np.argmax(outputs_start[row_i]))
            b = int(np.argmax(outputs_end[row_i]))

            if a > b:
                selected_text = tweet
            else:
                input_ids_row = batch_ids[row_i].tolist()
                mask_row = batch_mask[row_i].tolist()
                a = max(0, min(a, MAX_LEN - 1))
                b = max(0, min(b, MAX_LEN - 1))
                if mask_row[a] == 0 or mask_row[b] == 0:
                    selected_text = tweet
                else:
                    span_ids = input_ids_row[a : b + 1]
                    selected_text = tokenizer.decode(
                        span_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    ).strip()

                    if selected_text.strip() == "":
                        selected_text = tweet
                    else:
                        last = (
                            selected_text.split()[-1]
                            if len(selected_text.split())
                            else ""
                        )
                        if (last in set(string.punctuation)) or (
                            last.lower() in stop_words
                        ):
                            selected_text = " ".join(selected_text.split()[:-1]).strip()
                            if selected_text == "":
                                selected_text = tweet

            final_output.append(selected_text)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/387765864.py in <cell line: 0>()
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

/tmp/ipykernel_55/1711986970.py in __getitem__(self, item)
     25         selected = self.selected_texts[item]
     26 
---> 27         enc = tokenizer(
     28             tweet,
     29             self._sentiment_text(sentiment),

NameError: name 'tokenizer' is not defined

## === cell 9
sub = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

if len(final_output) != len(sub):
    raise RuntimeError(
        f"Prediction length mismatch: {len(final_output)} vs sample_submission {len(sub)}"
    )

sub.loc[:, "selected_text"] = final_output
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2939326765.py in <cell line: 0>()
      2 
      3 if len(final_output) != len(sub):
----> 4     raise RuntimeError(
      5         f"Prediction length mismatch: {len(final_output)} vs sample_submission {len(sub)}"
      6     )

RuntimeError: Prediction length mismatch: 0 vs sample_submission 2749

## === cell 10
sub.sample(20, random_state=0)
