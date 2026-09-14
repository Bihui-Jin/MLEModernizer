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

0.7193046808242798

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.4847) has done: 'I fix the model/config construction error by avoiding `RobertaConfig.from_pretrained(...)` (which is triggering a protobuf-related AttributeError in this environment) and instead using `AutoConfig.from_pretrained(...)` with `output_hidden_states=True`. I also make the inference robust to the missing external weights directory by falling back to a deterministic, competition-valid baseline prediction (return full text for neutral, first token for positive, last token for negative), so the notebook always produces `submission.csv`. These changes preserve the existing core model/prediction logic when weights are present, while ensuring end-to-end execution and a valid submission file when they are not. Finally, I keep paths unchanged and ensure the submission columns/quoting are correct via pandas CSV writing.'
- What this solution (achieved 0.33128) has done: 'I fix the runtime crash in model/config creation by avoiding the protobuf-triggering `AutoConfig.from_pretrained(...)` call in this environment and instead constructing a local `RobertaConfig` (no network/protobuf dependency) with `output_hidden_states=True`. This preserves the exact model architecture and forward logic, while allowing the script to run end-to-end deterministically. Since your current score (0.4847) is far below the target (~0.7193), I also remove the always-on heuristic fallback by letting the model run with randomly initialized weights when external checkpoints are missing; this is score-neutral-to-slightly-better than the heuristic in many cases while still being legitimate and minimal (no new training, no new data). The submission writing is kept intact and always produce a valid `submission.csv`.'
- What this solution (achieved 0.33128) has done: 'I fix the runtime crash caused by `RobertaConfig.from_pretrained("roberta-base")`, which in this environment triggers a protobuf `MessageFactory.GetPrototype` AttributeError. To keep the model architecture and forward pass identical (still RoBERTa-base sized with `output_hidden_states=True`), I construct a local `RobertaConfig` directly instead of calling `from_pretrained`. I also make the forward pass robust to models that don’t accept `token_type_ids` (RoBERTa ignores them), avoiding potential runtime errors, while preserving evaluation semantics. This should run end-to-end and produce a valid `submission.csv`; with the model now able to execute, the score should move up from the broken/degenerate behavior toward the target.'
- What this solution (achieved 0.4847) has done: 'I fix the runtime crash by removing the `torch.hub.load("pytorch/fairseq", ...)` dependency (it requires `hydra-core`, which isn’t installed) and replacing it with a local `transformers` RoBERTa backbone that still returns `hidden_states` as your model expects. This keeps the core architecture/head logic intact (same CNN + heads + hidden-state concatenation), while making model construction and inference run reliably offline in the Kaggle environment. I also make the forward call robust to RoBERTa ignoring `token_type_ids`, and ensure `word_preds` is always defined so `submission.csv` is written even if weights are missing or inference errors occur. These changes are execution-unblocking and should move the score above the heuristic baseline when weights are available; otherwise it still produce a valid submission.'
- What this solution (achieved 0.59324) has done: 'The crash comes from `AutoConfig.from_pretrained(...)` / `AutoModel.from_pretrained(...)` triggering a protobuf `MessageFactory.GetPrototype` AttributeError in this Kaggle image, so I remove that dependency by constructing a local RoBERTa config and initializing the model from config only (no protobuf/network). This keeps your core model architecture (RoBERTa-base-sized backbone with hidden states + the same CNN/heads) and preserves the existing inference logic, while ensuring the notebook runs end-to-end deterministically. Since your current score is far below the target, I also disable the “first/last token” heuristic fallback and instead use a more competitive, still-minimal baseline when no external checkpoints exist: return the full text for neutral, otherwise return the full text as well (a known strong baseline for this competition). The script always write a valid `submission.csv` with the required columns.'

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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel




## === cell 1
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
def load_model(model: nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)


def ensemble(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, df):
    whole = np.mean(np.stack(all_whole_preds, axis=0), axis=0)

    n = len(df)
    start_list, end_list, inst_list = [], [], []
    for i in range(n):
        s = torch.stack(
            [all_start_preds[f][i] for f in range(len(all_start_preds))], dim=0
        ).mean(dim=0)
        e = torch.stack(
            [all_end_preds[f][i] for f in range(len(all_end_preds))], dim=0
        ).mean(dim=0)
        inst = torch.stack(
            [all_inst_preds[f][i] for f in range(len(all_inst_preds))], dim=0
        ).mean(dim=0)
        start_list.append(s)
        end_list.append(e)
        inst_list.append(inst)
    return whole, start_list, end_list, inst_list


def _clean_text(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def _get_selected_text_from_offsets(
    text: str, offsets: List[Tuple[int, int]], start_idx: int, end_idx: int
) -> str:
    text = _clean_text(text)
    if len(offsets) == 0:
        return text
    start_idx = int(max(0, min(start_idx, len(offsets) - 1)))
    end_idx = int(max(0, min(end_idx, len(offsets) - 1)))
    if end_idx < start_idx:
        end_idx = start_idx
    s_char = offsets[start_idx][0]
    e_char = offsets[end_idx][1]
    if s_char is None or e_char is None:
        return text
    return text[s_char:e_char].strip()


def get_predicts_from_token_logits(
    all_whole_pred, all_start_pred, all_end_pred, all_inst_out, df, args
):
    word_preds = []
    inst_word_preds = []
    scores = []

    for i in range(len(df)):
        offsets = df.loc[i, "offsets"]
        text = df.loc[i, "text"]

        sp = all_start_pred[i].float()
        ep = all_end_pred[i].float()

        joint = sp.unsqueeze(1) * ep.unsqueeze(0)
        joint = torch.triu(joint)  # enforce e>=s
        flat_idx = torch.argmax(joint).item()
        s_idx = flat_idx // joint.size(1)
        e_idx = flat_idx % joint.size(1)

        s_tok = s_idx + args.offset
        e_tok = e_idx + args.offset

        pred = _get_selected_text_from_offsets(text, offsets, s_tok, e_tok)
        if pred == "":
            pred = _clean_text(text).strip()

        word_preds.append(pred)
        inst_word_preds.append(pred)
        scores.append(0.0)

    return word_preds, inst_word_preds, scores


def _baseline_selected_text(text: str, sentiment: str) -> str:
    """
    Minimal, competitive fallback baseline for this competition:
    - Neutral: full text
    - Positive/Negative: also full text (commonly strong baseline for Jaccard here)
    """
    text = _clean_text(text).strip()
    if text == "":
        return ""
    return text




## === cell 3
class TrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        df2=None,
        tokenizer=None,
        mode: str = "train",
        offset: int = 4,
        max_len: int = 128,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.mode = mode
        self.offset = offset
        self.max_len = max_len

        offsets_all = []
        input_ids_all = []
        attn_all = []
        type_ids_all = []

        for i in range(len(self.df)):
            text = _clean_text(self.df.loc[i, "text"])
            sentiment = _clean_text(self.df.loc[i, "sentiment"])
            enc = self.tokenizer(
                text,
                text_pair=sentiment,
                add_special_tokens=True,
                return_offsets_mapping=True,
                max_length=self.max_len,
                padding=False,
                truncation=True,
            )
            offsets = enc.get("offset_mapping", [])
            offsets_all.append(offsets)
            input_ids_all.append(enc["input_ids"])
            attn_all.append(enc["attention_mask"])
            type_ids_all.append(enc.get("token_type_ids", [0] * len(enc["input_ids"])))

        self.df["offsets"] = offsets_all
        self.input_ids_all = input_ids_all
        self.attn_all = attn_all
        self.type_ids_all = type_ids_all

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        input_ids = torch.tensor(self.input_ids_all[idx], dtype=torch.long)
        attn = torch.tensor(self.attn_all[idx], dtype=torch.long)
        ttype = torch.tensor(self.type_ids_all[idx], dtype=torch.long)

        dummy = torch.tensor(0, dtype=torch.long)
        return input_ids, ttype, attn, dummy, dummy, dummy, dummy, dummy, dummy, dummy


class MyCollator:
    def __call__(self, batch):
        tokens = [b[0] for b in batch]
        types = [b[1] for b in batch]
        masks = [b[2] for b in batch]
        rest = [b[3:] for b in batch]  # 7 dummies

        tokens = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, padding_value=1
        )  # RoBERTa pad id = 1
        types = torch.nn.utils.rnn.pad_sequence(
            types, batch_first=True, padding_value=0
        )
        masks = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0
        )

        dummies = []
        for j in range(len(rest[0])):
            dummies.append(torch.stack([r[j] for r in rest], dim=0))
        return (tokens, types, masks, *dummies)




## === cell 4
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")



## === cell 5
tokenizer = AutoTokenizer.from_pretrained(
    "roberta-base", use_fast=True, local_files_only=True
)




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
/tmp/ipykernel_55/3623716717.py in <cell line: 0>()
----> 1 tokenizer = AutoTokenizer.from_pretrained(
      2     "roberta-base", use_fast=True, local_files_only=True
      3 )
      4 
      5 

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1001                     config = AutoConfig.for_model(**config_dict)
   1002                 else:
-> 1003                     config = AutoConfig.from_pretrained(
   1004                         pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
   1005                     )

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1195         code_revision = kwargs.pop("code_revision", None)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
   1199         has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING

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
class Args:
    post = False
    tokenizer = tokenizer
    offset = 4
    batch_size = 16
    workers = 1


args = Args()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1849993304.py in <cell line: 0>()
----> 1 class Args:
      2     post = False
      3     tokenizer = tokenizer
      4     offset = 4
      5     batch_size = 16

/tmp/ipykernel_55/1849993304.py in Args()
      1 class Args:
      2     post = False
----> 3     tokenizer = tokenizer
      4     offset = 4
      5     batch_size = 16

NameError: name 'tokenizer' is not defined

## === cell 7
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
test = test_set.df




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/621728725.py in <cell line: 0>()
      1 collator = MyCollator()
      2 test_set = TrainDataset(
----> 3     test, None, tokenizer=tokenizer, mode="test", offset=args.offset
      4 )
      5 test_loader = DataLoader(

NameError: name 'tokenizer' is not defined

## === cell 8
class _HFBackboneWrapper(nn.Module):
    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        cfg = AutoConfig.from_pretrained(
            model_name,
            output_hidden_states=True,
            return_dict=True,
            local_files_only=True,
        )
        self.roberta = AutoModel.from_pretrained(
            model_name, config=cfg, local_files_only=True
        )
        self.config = self.roberta.config

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None, inputs_embeds=None
    ):
        return self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )


class TweetModel(nn.Module):
    def __init__(self, pretrain_path=None, dropout=0.2, config=None):
        super(TweetModel, self).__init__()
        self.bert = _HFBackboneWrapper()

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
            input_ids=inputs,
            attention_mask=masks,
            token_type_ids=token_type_ids,
            inputs_embeds=input_emb,
        )
        hs = out.hidden_states  # tuple(layer outputs)

        seq_output = torch.cat([hs[-1], hs[-2], hs[-3]], dim=-1)

        avg_output = torch.sum(seq_output * masks.unsqueeze(-1), dim=1, keepdim=False)
        avg_output = avg_output / torch.sum(masks, dim=-1, keepdim=True).clamp(min=1)
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0, 2, 1)).permute(0, 2, 1))

        se_out = self.se_head(self.dropout(seq_output))
        inst_out = self.inst_head(self.dropout(seq_output))
        return whole_out, se_out[:, :, 0], se_out[:, :, 1], inst_out




## === cell 9
def predict(
    model: nn.Module, valid_df, valid_loader, args, progress=False
) -> Dict[str, float]:
    model.eval()
    all_end_pred, all_whole_pred, all_start_pred, all_inst_out = [], [], [], []
    tq = None
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for tokens, types, masks, _, _, _, _, _, _, _ in valid_loader:
            if progress:
                tq.update(tokens.size(0))
            tokens = tokens.to(device)
            types = types.to(device)
            masks = masks.to(device)

            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)

            all_whole_pred.append(
                torch.softmax(whole_out, dim=-1)[:, 1].detach().cpu().numpy()
            )
            inst_out = torch.softmax(inst_out, dim=-1)

            for idx in range(len(start_out)):
                length = int(torch.sum(masks[idx, :]).item()) - 1  # -1 for last token
                length = max(length, args.offset + 1)

                all_start_pred.append(
                    torch.softmax(start_out[idx, args.offset : length], dim=-1)
                    .detach()
                    .cpu()
                )
                all_end_pred.append(
                    torch.softmax(end_out[idx, args.offset : length], dim=-1)
                    .detach()
                    .cpu()
                )
                all_inst_out.append(inst_out[idx, :, 1].detach().cpu())

    all_whole_pred = (
        np.concatenate(all_whole_pred, axis=0) if len(all_whole_pred) else np.array([])
    )
    if progress and tq is not None:
        tq.close()
    return all_whole_pred, all_start_pred, all_end_pred, all_inst_out




## === cell 10
model = TweetModel(config=None).to(device)

weights_dir = "/kaggle/input/roberta-weights-v10"
use_weights = os.path.isdir(weights_dir)

word_preds = None
try:
    if use_weights:
        all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = [], [], [], []
        for fold in range(5):
            ckpt = os.path.join(weights_dir, f"best-model-{fold}.pt")
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

            load_model(model, ckpt)
            model.to(device)

            fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = (
                predict(model, test, test_loader, args, progress=True)
            )

            all_whole_preds.append(fold_whole_preds)
            all_start_preds.append(fold_start_preds)
            all_end_preds.append(fold_end_preds)
            all_inst_preds.append(fold_inst_preds)

        all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = ensemble(
            all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test
        )

        word_preds, inst_word_preds, scores = get_predicts_from_token_logits(
            all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test, args
        )
    else:
        word_preds = [
            _baseline_selected_text(t, s)
            for t, s in zip(test["text"], test["sentiment"])
        ]
except Exception as e:
    print("Inference failed, using baseline fallback. Error:", repr(e))
    word_preds = [
        _baseline_selected_text(t, s) for t, s in zip(test["text"], test["sentiment"])
    ]

if word_preds is None or len(word_preds) != len(test):
    word_preds = [
        _baseline_selected_text(t, s) for t, s in zip(test["text"], test["sentiment"])
    ]



## --- ERROR in cell 10, traceback:
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
/tmp/ipykernel_55/3013662922.py in <cell line: 0>()
----> 1 model = TweetModel(config=None).to(device)
      2 
      3 weights_dir = "/kaggle/input/roberta-weights-v10"
      4 use_weights = os.path.isdir(weights_dir)
      5 

/tmp/ipykernel_55/1035138013.py in __init__(self, pretrain_path, dropout, config)
     29     def __init__(self, pretrain_path=None, dropout=0.2, config=None):
     30         super(TweetModel, self).__init__()
---> 31         self.bert = _HFBackboneWrapper()
     32 
     33         self.cnn = nn.Conv1d(

/tmp/ipykernel_55/1035138013.py in __init__(self, model_name)
      4         # Fix: avoid direct roberta.* imports that crash due to protobuf; use Auto* with local files only.
      5         # Score: use pretrained local roberta-base weights (vs random init) while keeping architecture identical.
----> 6         cfg = AutoConfig.from_pretrained(
      7             model_name,
      8             output_hidden_states=True,

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1195         code_revision = kwargs.pop("code_revision", None)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
   1199         has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING

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

## === cell 11
sub = pd.DataFrame(
    {
        "textID": test["textID"].astype(str),
        "selected_text": pd.Series(word_preds).astype(str),
    }
)
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/941611485.py in <cell line: 0>()
      2     {
      3         "textID": test["textID"].astype(str),
----> 4         "selected_text": pd.Series(word_preds).astype(str),
      5     }
      6 )

NameError: name 'word_preds' is not defined
