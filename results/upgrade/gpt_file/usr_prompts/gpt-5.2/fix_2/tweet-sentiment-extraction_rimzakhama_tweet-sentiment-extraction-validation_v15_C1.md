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

0.6186519861221313

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk("/kaggle/input"):
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

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class config:
    MAX_LEN = 141
    TRAIN_BATCH_SIZE = 40
    VALID_BATCH_SIZE = 16
    EPOCHS = 10

    BERT_PATH = "bert-base-uncased"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "/kaggle/input/tweet-sentiment-extraction/train.csv"

    TOKENIZER = None


try:
    from transformers.utils import cached_file

    vocab_path = cached_file(config.BERT_PATH, "vocab.txt", local_files_only=True)
    config.TOKENIZER = tokenizers.BertWordPieceTokenizer(vocab_path, lowercase=True)
except Exception:
    hf_tok = transformers.BertTokenizerFast.from_pretrained(
        config.BERT_PATH, local_files_only=True
    )
    config.TOKENIZER = hf_tok._tokenizer  # tokenizers.Tokenizer instance


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            config.BERT_PATH, local_files_only=True
        )
        self.l0 = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        sequence_output = out.last_hidden_state
        logits = self.l0(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BERTBaseUncased().to(device)
model = nn.DataParallel(model)

weights_path = "/kaggle/input/tweet-sentiment-extraction-training/model.bin"
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
else:
    print(
        f"WARNING: pretrained competition weights not found at {weights_path}. Using base bert weights."
    )

model.eval()




## --- ERROR in cell 1, traceback:
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
/tmp/ipykernel_55/1670703672.py in <cell line: 0>()
     40 
---> 41     vocab_path = cached_file(config.BERT_PATH, "vocab.txt", local_files_only=True)
     42     config.TOKENIZER = tokenizers.BertWordPieceTokenizer(vocab_path, lowercase=True)

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1670703672.py in <cell line: 0>()
     44     # Fallback: use transformers tokenizer and emulate needed fields via its backend if available.
     45     # (This keeps execution working if vocab.txt can't be located explicitly.)
---> 46     hf_tok = transformers.BertTokenizerFast.from_pretrained(
     47         config.BERT_PATH, local_files_only=True
     48     )

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

/usr/local/lib/python3.11/dist-packages/transformers/models/bert/tokenization_bert.py in __init__(self, vocab_file, do_lower_case, do_basic_tokenize, never_split, unk_token, sep_token, pad_token, cls_token, mask_token, tokenize_chinese_chars, strip_accents, clean_up_tokenization_spaces, **kwargs)
    112         **kwargs,
    113     ):
--> 114         if not os.path.isfile(vocab_file):
    115             raise ValueError(
    116                 f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"

/usr/lib/python3.11/genericpath.py in isfile(path)

TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType

## === cell 2
class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = str(self.tweet[item])
        tweet = " ".join(tweet.split())

        selected_text = str(self.selected_text[item])
        selected_text = " ".join(selected_text.split())

        len_sel_text = len(selected_text)

        idx0 = -1
        idx1 = -1
        if len_sel_text > 0 and len(tweet) > 0:
            for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
                if tweet[ind : ind + len_sel_text] == selected_text:
                    idx0 = ind
                    idx1 = ind + len_sel_text - 1
                    break

        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                if tweet[j] != " ":
                    char_targets[j] = 1

        tok_tweet = self.tokenizer.encode(tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids

        tok_tweet_offsets = tok_tweet.offsets[1:-1]

        targets = [0] * (len(tok_tweet_tokens) - 2)
        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                targets[j] = 1

        targets = [0] + targets + [0]
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1

        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * len(tok_tweet_ids)

        padding_len = self.max_len - len(tok_tweet_ids)
        if padding_len < 0:
            tok_tweet_ids = tok_tweet_ids[: self.max_len]
            mask = mask[: self.max_len]
            token_type_ids = token_type_ids[: self.max_len]
            targets = targets[: self.max_len]
            targets_start = targets_start[: self.max_len]
            targets_end = targets_end[: self.max_len]
            padding_len = 0
            tok_tweet_tokens = tok_tweet_tokens[: self.max_len]
        else:
            ids = tok_tweet_ids + [0] * padding_len
            mask = mask + [0] * padding_len
            token_type_ids = token_type_ids + [0] * padding_len
            targets = targets + [0] * padding_len
            targets_start = targets_start + [0] * padding_len
            targets_end = targets_end + [0] * padding_len
            tok_tweet_ids = ids

        sentiment = [1, 0, 0]
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            "ids": torch.tensor(tok_tweet_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "targets_start": torch.tensor(targets_start, dtype=torch.long),
            "targets_end": torch.tensor(targets_end, dtype=torch.long),
            "padding_len": torch.tensor(padding_len, dtype=torch.long),
            "tweet_tokens": " ".join(tok_tweet_tokens),
            "orig_tweet": self.tweet[item],
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_sentiment": self.sentiment[item],
            "orig_selected_text": self.selected_text[item],
        }




## === cell 3
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values

test_dataset = TweetDataset(
    tweet=df_test.text.values,
    sentiment=df_test.sentiment.values,
    selected_text=df_test.selected_text.values,
)

valid_data_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=False, batch_size=config.VALID_BATCH_SIZE, num_workers=1
)

final_col = []
fin_output_start = []
fin_output_end = []
fin_padding_lens = []
fin_tweet_tokens = []
fin_orig_sentiment = []
fin_orig_selected_text = []
fin_orig_tweet = []

with torch.no_grad():
    for bi, d in enumerate(valid_data_loader):
        ids = d["ids"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)

        tweet_tokens = d["tweet_tokens"]
        padding_len = d["padding_len"]
        orig_sentiment = d["orig_sentiment"]
        orig_selected_text = d["orig_selected_text"]
        orig_tweet = d["orig_tweet"]

        o1, o2 = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        fin_output_start.append(torch.sigmoid(o1).cpu().detach().numpy())
        fin_output_end.append(torch.sigmoid(o2).cpu().detach().numpy())
        fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())

        fin_tweet_tokens.extend(tweet_tokens)
        fin_orig_sentiment.extend(list(orig_sentiment))
        fin_orig_selected_text.extend(list(orig_selected_text))
        fin_orig_tweet.extend(list(orig_tweet))

fin_output_start = np.vstack(fin_output_start)
fin_output_end = np.vstack(fin_output_end)

threshold = 0.2
for j in range(len(fin_tweet_tokens)):
    tweet_tokens = fin_tweet_tokens[j]
    padding_len = fin_padding_lens[j]
    original_tweet = fin_orig_tweet[j]
    sentiment = fin_orig_sentiment[j]

    if padding_len > 0:
        mask_start = fin_output_start[j, :][:-padding_len] >= threshold
        mask_end = fin_output_end[j, :][:-padding_len] >= threshold
    else:
        mask_start = fin_output_start[j, :] >= threshold
        mask_end = fin_output_end[j, :] >= threshold

    mask = [0] * len(mask_start)
    idx_start = np.nonzero(mask_start)[0]
    idx_end = np.nonzero(mask_end)[0]

    if len(idx_start) > 0:
        idx_start = int(idx_start[0])
        if len(idx_end) > 0:
            idx_end = int(idx_end[0])
        else:
            idx_end = idx_start
    else:
        idx_start = 0
        idx_end = 0

    if idx_end < idx_start:
        idx_end = idx_start

    for mj in range(idx_start, min(idx_end + 1, len(mask))):
        mask[mj] = 1

    output_tokens = [
        x for p, x in enumerate(tweet_tokens.split()) if p < len(mask) and mask[p] == 1
    ]
    output_tokens = [x for x in output_tokens if x not in ("[CLS]", "[SEP]")]

    final_output = ""
    for ot in output_tokens:
        if ot.startswith("##"):
            final_output = final_output + ot[2:]
        elif len(ot) == 1 and ot in string.punctuation:
            final_output = final_output + ot
        else:
            final_output = final_output + " " + ot
    final_output = final_output.strip()

    if sentiment == "neutral" or len(str(original_tweet).split()) < 4:
        final_output = str(original_tweet)

    final_col.append(final_output)

submission = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)
submission["selected_text"] = final_col

assert len(submission) == len(
    df_test
), f"Submission rows {len(submission)} != test rows {len(df_test)}"
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1123690599.py in <cell line: 0>()
     22 
     23 with torch.no_grad():
---> 24     for bi, d in enumerate(valid_data_loader):
     25         ids = d["ids"].to(device, dtype=torch.long)
     26         token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)

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

AttributeError: Caught AttributeError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/288786241.py", line 36, in __getitem__
    tok_tweet = self.tokenizer.encode(tweet)
                ^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'encode'
