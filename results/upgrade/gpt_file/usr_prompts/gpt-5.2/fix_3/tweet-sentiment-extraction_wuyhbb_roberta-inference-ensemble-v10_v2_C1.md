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

0.7257987260818481

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
import random
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    RobertaConfig,
    RobertaTokenizerFast,
    AutoConfig,
    AutoModel,
)




## === cell 1
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model: nn.Module, path: str, map_location: str = "cpu") -> None:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if len(unexpected) > 50:
        print("Warning: many unexpected keys while loading:", len(unexpected))


def ensemble(
    all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, df, softmax=True
):
    whole = np.mean(np.stack(all_whole_preds, axis=0), axis=0)

    n = len(df)
    start_list, end_list, inst_list = [], [], []
    for i in range(n):
        s = torch.stack(
            [all_start_preds[f][i].float() for f in range(len(all_start_preds))], dim=0
        ).mean(0)
        e = torch.stack(
            [all_end_preds[f][i].float() for f in range(len(all_end_preds))], dim=0
        ).mean(0)
        ins = torch.stack(
            [all_inst_preds[f][i].float() for f in range(len(all_inst_preds))], dim=0
        ).mean(0)
        start_list.append(s)
        end_list.append(e)
        inst_list.append(ins)
    return whole, start_list, end_list, inst_list


def _clean_text_basic(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return " ".join(str(x).split())


def _select_from_offsets(
    text: str, offsets: List[Tuple[int, int]], start_idx: int, end_idx: int
) -> str:
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    start_idx = max(0, min(start_idx, len(offsets) - 1))
    end_idx = max(0, min(end_idx, len(offsets) - 1))

    spans = [
        (s, e)
        for (s, e) in offsets[start_idx : end_idx + 1]
        if not (s == 0 and e == 0) and e >= s
    ]
    if not spans:
        return _clean_text_basic(text)
    s_char = min(s for s, _ in spans)
    e_char = max(e for _, e in spans)
    if e_char <= s_char:
        return _clean_text_basic(text)
    return text[s_char:e_char]


def get_predicts_from_token_logits(
    whole_pred: np.ndarray,
    start_pred: List[torch.Tensor],
    end_pred: List[torch.Tensor],
    inst_pred: List[torch.Tensor],
    df: pd.DataFrame,
    args,
):
    word_preds, inst_word_preds, scores = [], [], []
    for i, row in df.iterrows():
        text = str(row["text"])
        sentiment = str(row["sentiment"])

        if sentiment == "neutral":
            pred = _clean_text_basic(text)
            word_preds.append(pred)
            inst_word_preds.append(pred)
            scores.append(0.0)
            continue

        s_logits = start_pred[i]
        e_logits = end_pred[i]

        s_idx = int(torch.argmax(s_logits).item())
        e_idx = int(torch.argmax(e_logits).item())

        offsets = df.at[i, "offsets"]
        pred = _select_from_offsets(
            text, offsets, s_idx - args.offset, e_idx - args.offset
        )
        pred = _clean_text_basic(pred)

        word_preds.append(pred)
        inst_word_preds.append(pred)
        scores.append(float(whole_pred[i]) if whole_pred is not None else 0.0)

    return word_preds, inst_word_preds, scores




## === cell 2
class TrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        labels=None,
        tokenizer=None,
        mode="train",
        offset=4,
        max_len=96,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode
        self.offset = offset
        self.max_len = max_len

        self.encodings = []
        self.offsets = []
        for _, row in self.df.iterrows():
            text = _clean_text_basic(row["text"])
            sentiment = str(row["sentiment"])
            sent_token = sentiment

            enc = self.tokenizer(
                sent_token,
                text,
                add_special_tokens=True,
                return_offsets_mapping=True,
                padding=False,
                truncation=True,
                max_length=self.max_len,
            )
            self.encodings.append(enc)
            self.offsets.append(enc["offset_mapping"])

        self.df["offsets"] = self.offsets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(
            enc.get("token_type_ids", [0] * len(enc["input_ids"])), dtype=torch.long
        )
        dummy = torch.tensor(0, dtype=torch.long)
        return (
            input_ids,
            token_type_ids,
            attention_mask,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
        )


class MyCollator:
    def __call__(self, batch):
        input_ids, token_type_ids, attention_mask, d1, d2, d3, d4, d5, d6, d7 = zip(
            *batch
        )
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=1
        )  # roberta pad id is 1
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        dummy = torch.stack(d1)
        return (
            input_ids,
            token_type_ids,
            attention_mask,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
        )




## === cell 3
set_seed(42)
DATA_DIR = "../input/tweet-sentiment-extraction"
if not os.path.exists(os.path.join(DATA_DIR, "test.csv")):
    DATA_DIR = "../kaggle/input/tweet-sentiment-extraction"

test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))




## === cell 4
def _find_local_roberta_dir() -> str:
    candidates = [
        "../input/roberta-base",
        "../kaggle/input/roberta-base",
        "../input/roberta-weights-v10/roberta-base",
        "../kaggle/input/roberta-weights-v10/roberta-base",
        "../input/roberta-weights-v10",
        "../kaggle/input/roberta-weights-v10",
    ]
    for p in candidates:
        if os.path.isdir(p):
            has_token_files = (
                os.path.exists(os.path.join(p, "merges.txt"))
                and os.path.exists(os.path.join(p, "vocab.json"))
            ) or os.path.exists(os.path.join(p, "tokenizer.json"))
            if has_token_files:
                return p
    return ""


ROBERTA_DIR = _find_local_roberta_dir()
if not ROBERTA_DIR:
    raise FileNotFoundError(
        "Could not find local RoBERTa tokenizer files (vocab/merges or tokenizer.json). "
        "Please add a roberta-base dataset or include tokenizer files inside roberta-weights-v10."
    )

tokenizer = RobertaTokenizerFast.from_pretrained(ROBERTA_DIR)


class Args:
    post = False
    tokenizer = tokenizer
    offset = 4
    batch_size = 16
    workers = 1


args = Args()

collator = MyCollator()
test_set = TrainDataset(
    test, None, tokenizer=tokenizer, mode="test", offset=args.offset
)
test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collator,
    num_workers=args.workers,
    pin_memory=torch.cuda.is_available(),
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/813542725.py in <cell line: 0>()
     24 ROBERTA_DIR = _find_local_roberta_dir()
     25 if not ROBERTA_DIR:
---> 26     raise FileNotFoundError(
     27         "Could not find local RoBERTa tokenizer files (vocab/merges or tokenizer.json). "
     28         "Please add a roberta-base dataset or include tokenizer files inside roberta-weights-v10."

FileNotFoundError: Could not find local RoBERTa tokenizer files (vocab/merges or tokenizer.json). Please add a roberta-base dataset or include tokenizer files inside roberta-weights-v10.

## === cell 5
class TweetModel(nn.Module):

    def __init__(self, pretrain_path=None, dropout=0.2, config=None):
        super(TweetModel, self).__init__()
        if config is not None:
            self.bert = AutoModel.from_config(config)
        else:
            config = AutoConfig.from_pretrained(
                pretrain_path, output_hidden_states=True
            )
            self.bert = AutoModel.from_pretrained(
                pretrain_path, cache_dir=None, config=config
            )

        self.cnn = nn.Conv1d(
            self.bert.config.hidden_size * 3, self.bert.config.hidden_size, 3, padding=1
        )
        self.gelu = nn.GELU()

        self.whole_head = nn.Sequential(
            OrderedDict(
                [
                    ("dropout1", nn.Dropout(0.1)),
                    ("l1", nn.Linear(self.bert.config.hidden_size * 3, 256)),
                    ("act1", nn.GELU()),
                    ("dropout2", nn.Dropout(0.1)),
                    ("l2", nn.Linear(256, 2)),
                ]
            )
        )
        self.se_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.inst_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, masks, token_type_ids=None, input_emb=None):
        out = self.bert(
            inputs,
            attention_mask=masks,
            token_type_ids=token_type_ids,
            inputs_embeds=input_emb,
        )
        if len(out) >= 3:
            last_hidden = out[0]
            pooled_output = out[1] if out[1] is not None else last_hidden[:, 0]
            hs = out[2]
        else:
            last_hidden = out[0]
            pooled_output = last_hidden[:, 0]
            hs = out.hidden_states

        seq_output = torch.cat([hs[-1], hs[-2], hs[-3]], dim=-1)

        avg_output = F.adaptive_avg_pool1d(seq_output.permute(0, 2, 1), 1).squeeze(-1)
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0, 2, 1)).permute(0, 2, 1))

        se_out = self.se_head(self.dropout(seq_output))
        inst_out = self.inst_head(self.dropout(seq_output))
        return whole_out, se_out[:, :, 0], se_out[:, :, 1], inst_out




## === cell 6
def predict(
    model: nn.Module, valid_df, valid_loader, args, device, progress=False
) -> Dict[str, float]:
    model.eval()
    all_end_pred, all_whole_pred, all_start_pred, all_inst_out = [], [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for tokens, types, masks, _, _, _, _, _, _, _ in valid_loader:
            if progress:
                batch_size = tokens.size(0)
                tq.update(batch_size)
            masks = masks.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            types = types.to(device, non_blocking=True)
            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)
            start_out = start_out.masked_fill(~masks.bool(), -1000)
            end_out = end_out.masked_fill(~masks.bool(), -1000)

            start_out = torch.softmax(start_out, dim=-1)
            end_out = torch.softmax(end_out, dim=-1)

            all_whole_pred.append(
                torch.softmax(whole_out, dim=-1)[:, 1].detach().cpu().numpy()
            )
            inst_out = torch.softmax(inst_out, dim=-1)
            for idx in range(len(start_out)):
                all_start_pred.append(start_out[idx, :].detach().cpu())
                all_end_pred.append(end_out[idx, :].detach().cpu())
                all_inst_out.append(inst_out[idx, :, 1].detach().cpu())
            assert all_start_pred[-1].dim() == 1

    all_whole_pred = np.concatenate(all_whole_pred)

    if progress:
        tq.close()
    return all_whole_pred, all_start_pred, all_end_pred, all_inst_out




## === cell 7
config = RobertaConfig.from_pretrained(ROBERTA_DIR, output_hidden_states=True)
model = TweetModel(config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## --- ERROR in cell 7, traceback:
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
    159     if not REPO_ID_REGEX.match(repo_id):
--> 160         raise HFValidationError(
    161             "Repo id must use alphanumeric chars, '-', '_' or '.'."

HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: ''.

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
    159     if not REPO_ID_REGEX.match(repo_id):
--> 160         raise HFValidationError(
    161             "Repo id must use alphanumeric chars, '-', '_' or '.'."

HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: ''.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/1094698377.py in <cell line: 0>()
      1 # Bugfix: config also needs to be loaded from local files (offline-safe).
----> 2 config = RobertaConfig.from_pretrained(ROBERTA_DIR, output_hidden_states=True)
      3 model = TweetModel(config=config)
      4 
      5 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

OSError: Can't load the configuration of ''. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '' is the correct path to a directory containing a config.json file

## === cell 8
all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = [], [], [], []

WEIGHTS_DIR = "../input/roberta-weights-v10"
if not os.path.exists(WEIGHTS_DIR):
    WEIGHTS_DIR = "../kaggle/input/roberta-weights-v10"

missing = [
    f
    for f in [f"best-model-{i}.pt" for i in range(5)]
    if not os.path.exists(os.path.join(WEIGHTS_DIR, f))
]
if missing:
    raise FileNotFoundError(
        f"Missing weight files in {WEIGHTS_DIR}: {missing}. "
        "Please attach the 'roberta-weights-v10' dataset (or correct WEIGHTS_DIR)."
    )

for fold in range(5):
    load_model(
        model, os.path.join(WEIGHTS_DIR, f"best-model-{fold}.pt"), map_location=device
    )
    model.to(device)
    fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = predict(
        model, test_set.df, test_loader, args, device=device, progress=True
    )
    all_whole_preds.append(fold_whole_preds)
    all_start_preds.append(fold_start_preds)
    all_end_preds.append(fold_end_preds)
    all_inst_preds.append(fold_inst_preds)

all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = ensemble(
    all_whole_preds,
    all_start_preds,
    all_end_preds,
    all_inst_preds,
    test_set.df,
    softmax=True,
)

word_preds, inst_word_preds, scores = get_predicts_from_token_logits(
    all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test_set.df, args
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1367206444.py in <cell line: 0>()
     12 ]
     13 if missing:
---> 14     raise FileNotFoundError(
     15         f"Missing weight files in {WEIGHTS_DIR}: {missing}. "
     16         "Please attach the 'roberta-weights-v10' dataset (or correct WEIGHTS_DIR)."

FileNotFoundError: Missing weight files in ../kaggle/input/roberta-weights-v10: ['best-model-0.pt', 'best-model-1.pt', 'best-model-2.pt', 'best-model-3.pt', 'best-model-4.pt']. Please attach the 'roberta-weights-v10' dataset (or correct WEIGHTS_DIR).

## === cell 9
test["selected_text"] = word_preds
sub = test[["textID", "selected_text"]].copy()
sub.to_csv("submission.csv", index=False)
print(sub.head(10))
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2921850188.py in <cell line: 0>()
----> 1 test["selected_text"] = word_preds
      2 sub = test[["textID", "selected_text"]].copy()
      3 sub.to_csv("submission.csv", index=False)
      4 print(sub.head(10))
      5 print("Wrote submission.csv with shape:", sub.shape)

NameError: name 'word_preds' is not defined
