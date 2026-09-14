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

0.7285080552101135

# 6. Current score

0.59324

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fix the runtime crash happening during model/tokenizer loading by pinning protobuf to the pure-Python implementation (a common Kaggle/transformers incompatibility that triggers `MessageFactory.GetPrototype` errors). I also make the checkpoint directory robust by searching the Kaggle input tree for the expected `best-model-*.pt` files, so the script doesn’t fail due to a slightly different dataset mount path. Finally, I ensure the pipeline always reaches submission writing by guarding against missing predictions and by producing a valid `submission.csv` with the required columns and quoting handled by pandas.'
- What this solution (achieved 0.59324) has done: 'The crash is coming from an incompatibility between the protobuf C++ implementation and the Transformers fast tokenizer/model loading in this environment; setting the env var inside Python is too late in some Kaggle kernels because `google.protobuf` may already be imported indirectly. I force the pure-Python protobuf implementation earlier and also disable Transformers’ fast-path that can trigger protobuf usage, while keeping the same model/weights/prediction logic. To nudge the score upward toward the target, I also fix a logic bug in `predict()` where a softmax is incorrectly applied to 1D start/end logits; using the correct 1D softmax preserves the intended semantics and typically improves span selection quality without changing the architecture/training. Finally, I keep the submission writing robust and ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 0.59324) has done: 'I fix the immediate runtime error by switching back to a Fast tokenizer (required for `return_offsets_mapping`) while keeping protobuf on the pure-Python implementation to avoid the `MessageFactory.GetPrototype` crash. To make that protobuf setting actually take effect before Transformers imports, I move the environment-variable setup to the very top of the script (before any `transformers`/`sentencepiece`-related imports). These changes are execution-critical and score-positive because they restore correct offset mapping (needed to reconstruct spans), rather than falling back to whole-text predictions. I keep the model, checkpoints, prediction logic, and submission-writing logic the same, only adjusting what’s required to run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved 0.59324) has done: 'I fix the protobuf-related crash by forcing the pure-Python protobuf implementation to be used before any transformer/protobuf-dependent imports take place, and by explicitly disabling Transformers’ fast-protobuf code path via environment variables. This is execution-critical and should be score-neutral (it doesn’t change the model or prediction logic), but it allow the script to run end-to-end and actually use the checkpoints for predictions instead of failing. I also make the checkpoint directory fallback robust to the expected Kaggle input mount and keep submission writing identical while ensuring the CSV is always produced. No architecture/training/prediction-core changes are made beyond making the environment compatible with the existing code.'
- What this solution (achieved 0.59324) has done: 'I fix the protobuf crash by forcing the pure-Python protobuf backend and disabling the C++/upb implementation *before* any transformer-related import happens (this is the direct cause of the `MessageFactory.GetPrototype` error). I also make that setting robust by purging any already-imported `google.protobuf` modules in case the notebook environment imported them earlier. These changes are execution-critical and score-neutral (they don’t change your model, checkpoints, or span logic), so your pipeline should run end-to-end and generate `submission.csv`. I keep your architecture/training/prediction logic intact and only touch the environment/bootstrap and import order.'
- What this solution (achieved 0.59324) has done: 'I fix the remaining protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf backend before any Transformers/protobuf-dependent imports, and by ensuring any already-imported `google.protobuf` modules are fully purged (including submodules) before importing `transformers`. This is execution-critical and should be score-neutral (same model/checkpoints/prediction logic), but it actually allow the 10-fold ensemble to run instead of crashing. I also make the input path resolution more robust by falling back to the known `/kaggle/input/...` locations without changing file names. Finally, I keep submission formatting identical and guaranteed to write `submission.csv` with the required columns.'
- What this solution (achieved 0.59324) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before any other imports and by also disabling the C++/upb backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3`, and then purging any already-imported protobuf modules. This is execution-critical and score-neutral (it doesn’t change the model/prediction logic), but it unblocks the full 10-fold inference so the submission is produced from the checkpoints instead of failing. I also make the checkpoint discovery slightly more robust by accepting any directory that contains at least the required fold files, while keeping the same fold-loop and averaging logic. Finally, I keep the submission writing unchanged and guaranteed to output a valid `submission.csv`.'
- What this solution (achieved 0.59324) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* any other imports and by also disabling the upb/C++ backend; additionally I purge any already-imported protobuf modules to ensure the setting takes effect in Kaggle. This is execution-critical and score-neutral (it doesn’t alter the model, checkpoints, or span-selection logic), but it allows the 10-fold inference to actually run instead of failing. I also make the input directory resolution consistent with the provided dataset paths so the script reliably finds `test.csv` and the checkpoint folder under `/kaggle/input`. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.59324) has done: 'I fix the protobuf crash by forcing the pure-Python protobuf implementation *at process start* (not after imports) and by purging any already-imported protobuf modules before importing `transformers`, which is the direct cause of the `MessageFactory.GetPrototype` error. I also make model construction avoid any protobuf-dependent code paths by instantiating RoBERTa directly from `RobertaConfig` (so no `Auto*` factory logic is needed), while keeping the exact same architecture/forward/prediction logic and checkpoints. These changes are execution-critical and score-positive because they let the script actually run the 10-fold ensemble instead of crashing and falling back to whole-text outputs. Submission writing stays the same and always produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_UPB", "1")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import sys

for m in list(sys.modules.keys()):
    if m == "google.protobuf" or m.startswith("google.protobuf."):
        del sys.modules[m]

import re
import json
import math
import random
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import tqdm

from transformers import AutoTokenizer, RobertaConfig, RobertaModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR = "/kaggle/input/tweet-sentiment-extraction"
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_model(model: nn.Module, path: str):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)
    return model


def ensemble(all_whole, all_start, all_end, all_inst, df):
    whole = np.mean(np.stack(all_whole, axis=0), axis=0)

    n = len(df)
    start_out, end_out, inst_out = [], [], []
    for i in range(n):
        start_i = torch.stack(
            [all_start[f][i] for f in range(len(all_start))], dim=0
        ).mean(dim=0)
        end_i = torch.stack([all_end[f][i] for f in range(len(all_end))], dim=0).mean(
            dim=0
        )
        inst_i = torch.stack(
            [all_inst[f][i] for f in range(len(all_inst))], dim=0
        ).mean(dim=0)
        start_out.append(start_i)
        end_out.append(end_i)
        inst_out.append(inst_i)
    return whole, start_out, end_out, inst_out


def _select_span_from_logits(
    start_probs: torch.Tensor, end_probs: torch.Tensor, max_len: int = 30
) -> Tuple[int, int]:
    s = start_probs.detach().cpu().numpy()
    e = end_probs.detach().cpu().numpy()
    best_i, best_j = 0, 0
    best = -1.0
    for i in range(len(s)):
        j_max = min(len(e) - 1, i + max_len)
        j = i + int(np.argmax(e[i : j_max + 1]))
        score = s[i] * e[j]
        if score > best:
            best = score
            best_i, best_j = i, j
    return best_i, best_j


def map_to_word(
    text: str, enc_offsets: List[Tuple[int, int]], start_idx: int, end_idx: int
) -> str:
    if not enc_offsets:
        return text
    start_idx = max(0, min(start_idx, len(enc_offsets) - 1))
    end_idx = max(0, min(end_idx, len(enc_offsets) - 1))
    if end_idx < start_idx:
        end_idx = start_idx
    char_s = enc_offsets[start_idx][0]
    char_e = enc_offsets[end_idx][1]
    if char_s is None or char_e is None:
        return text
    char_s = max(0, min(char_s, len(text)))
    char_e = max(0, min(char_e, len(text)))
    out = text[char_s:char_e].strip()
    return out if out else text.strip()


def get_predicts_from_token_logits(
    whole_pred, start_preds, end_preds, inst_preds, df, args
):
    word_preds = []
    inst_word_preds = []
    scores = []
    for i in range(len(df)):
        text = str(df.loc[i, "text"])
        offsets = df.loc[i, "_offsets"]

        if str(df.loc[i, "sentiment"]) == "neutral":
            pred = text.strip()
            word_preds.append(pred)
            inst_word_preds.append(pred)
            scores.append(1.0)
            continue

        si, ei = _select_span_from_logits(start_preds[i], end_preds[i], max_len=30)
        pred = map_to_word(text, offsets, si, ei)

        inst_pred = pred
        if not inst_pred.strip():
            inst_pred = text.strip()

        word_preds.append(pred)
        inst_word_preds.append(inst_pred)
        scores.append(float(whole_pred[i]))
    return word_preds, inst_word_preds, scores


class TrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        _y=None,
        tokenizer=None,
        mode="test",
        offset=4,
        max_len=128,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode
        self.offset = offset
        self.max_len = max_len

        self.encodings = []
        for i in range(len(self.df)):
            text = str(self.df.loc[i, "text"])
            sentiment = str(self.df.loc[i, "sentiment"])
            enc = self.tokenizer(
                sentiment,
                text,
                add_special_tokens=True,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_len,
                padding=False,
            )
            self.encodings.append(enc)
        self.df["_offsets"] = [enc["offset_mapping"] for enc in self.encodings]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attn = torch.tensor(enc["attention_mask"], dtype=torch.long)
        token_type = torch.zeros_like(input_ids)

        return input_ids, token_type, attn, None, None, None, None, None, None, None


class MyCollator:
    def __call__(self, batch):
        input_ids, token_type_ids, attention_masks = [], [], []
        for b in batch:
            input_ids.append(b[0])
            token_type_ids.append(b[1])
            attention_masks.append(b[2])
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=1
        )  # roberta pad token id=1
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attention_masks = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        return (
            input_ids,
            token_type_ids,
            attention_masks,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )




## === cell 2
if not os.path.exists(TEST_PATH):
    candidates = [
        "/kaggle/input/tweet-sentiment-extraction/test.csv",
        "/kaggle/input/test.csv",
        "../input/test.csv",
        "/kaggle/input/tweet-sentiment-extraction/tweet-sentiment-extraction/test.csv",
        "../input/tweet-sentiment-extraction/tweet-sentiment-extraction/test.csv",
        "/kaggle/data/test.csv",
        "/kaggle/data/tweet-sentiment-extraction/test.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            TEST_PATH = c
            break

test = pd.read_csv(TEST_PATH)
test["text"] = test["text"].fillna("").astype(str)
test["sentiment"] = test["sentiment"].fillna("neutral").astype(str)

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)


class Args:
    post = True
    tokenizer = tokenizer
    offset = 4
    batch_size = 32
    workers = 0  # safer in Kaggle notebooks


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




## === cell 3
class TweetModel(nn.Module):

    def __init__(self, pretrain_path=None, dropout=0.2, config=None):
        super(TweetModel, self).__init__()

        if config is None:
            config = RobertaConfig.from_pretrained(
                "roberta-base", output_hidden_states=True
            )
        else:
            config.output_hidden_states = True

        self.bert = RobertaModel(config)

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
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states

        seq_output = torch.cat([hs[-1], hs[-2], hs[-3]], dim=-1)

        avg_output = torch.sum(seq_output * masks.unsqueeze(-1), dim=1, keepdim=False)
        avg_output = avg_output / torch.sum(masks, dim=-1, keepdim=True).clamp(min=1)
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0, 2, 1)).permute(0, 2, 1))

        se_out = self.se_head(self.dropout(seq_output))
        inst_out = self.inst_head(self.dropout(seq_output))
        return whole_out, se_out[:, :, 0], se_out[:, :, 1], inst_out


def predict(model: nn.Module, valid_df, valid_loader, args, progress=False):
    model.eval()
    all_end_pred, all_whole_pred, all_start_pred, all_inst_out = [], [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for tokens, types, masks, _, _, _, _, _, _, _ in valid_loader:
            if progress:
                tq.update(tokens.size(0))
            tokens = tokens.to(DEVICE)
            types = types.to(DEVICE)
            masks = masks.to(DEVICE)

            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)

            all_whole_pred.append(
                torch.softmax(whole_out, dim=-1)[:, 1].detach().cpu().numpy()
            )
            inst_out = torch.softmax(inst_out, dim=-1)

            for idx in range(len(start_out)):
                length = int(torch.sum(masks[idx, :]).item()) - 1  # -1 for last token
                length = max(length, args.offset + 1)

                start_slice = start_out[idx, args.offset : length]
                end_slice = end_out[idx, args.offset : length]

                start_prob = torch.softmax(start_slice, dim=0).detach().cpu()
                end_prob = torch.softmax(end_slice, dim=0).detach().cpu()

                all_start_pred.append(start_prob)
                all_end_pred.append(end_prob)
                all_inst_out.append(inst_out[idx, :, 1].detach().cpu())
    if progress:
        tq.close()
    all_whole_pred = np.concatenate(all_whole_pred, axis=0)
    return all_whole_pred, all_start_pred, all_end_pred, all_inst_out




## === cell 4
config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)
model = TweetModel(config=config).to(DEVICE)

all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = [], [], [], []

CKPT_DIR = "../input/roberta-v10-10"


def find_ckpt_dir(preferred_dir: str, needed_files: int = 10) -> str:
    def has_needed(dirpath: str) -> bool:
        if not os.path.isdir(dirpath):
            return False
        for fold in range(needed_files):
            if not os.path.exists(os.path.join(dirpath, f"best-model-{fold}.pt")):
                return False
        return True

    if has_needed(preferred_dir):
        return preferred_dir

    for root in ("/kaggle/input", "../input", "/kaggle/data"):
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                if any(
                    fn.startswith("best-model-") and fn.endswith(".pt")
                    for fn in filenames
                ):
                    ok = True
                    for fold in range(needed_files):
                        if f"best-model-{fold}.pt" not in filenames:
                            ok = False
                            break
                    if ok:
                        return dirpath
    return preferred_dir


CKPT_DIR = find_ckpt_dir(CKPT_DIR, needed_files=10)

for fold in range(10):
    ckpt_path = os.path.join(CKPT_DIR, f"best-model-{fold}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path} (CKPT_DIR={CKPT_DIR})"
        )
    load_model(model, ckpt_path)
    model.to(DEVICE)

    fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = predict(
        model, test, test_loader, args, progress=True
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



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1820165583.py in <cell line: 0>()
     41     ckpt_path = os.path.join(CKPT_DIR, f"best-model-{fold}.pt")
     42     if not os.path.exists(ckpt_path):
---> 43         raise FileNotFoundError(
     44             f"Missing checkpoint: {ckpt_path} (CKPT_DIR={CKPT_DIR})"
     45         )

FileNotFoundError: Missing checkpoint: ../input/roberta-v10-10/best-model-0.pt (CKPT_DIR=../input/roberta-v10-10)

## === cell 5
if "word_preds" not in globals() or word_preds is None or len(word_preds) != len(test):
    word_preds = test["text"].fillna("").astype(str).str.strip().tolist()

test["selected_text"] = [str(x) if isinstance(x, str) else "" for x in word_preds]
sub = test[["textID", "selected_text"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
