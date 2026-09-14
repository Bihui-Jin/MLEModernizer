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

0.7258532643318176

# 6. Current score

0.59324

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fix the crash when creating the Hugging Face config/model by avoiding the Protobuf-dependent path that triggers the `MessageFactory` error in this environment, and instead load the model with `from_pretrained(..., output_hidden_states=True)` which preserves the same core architecture/forward semantics. I also fix the missing checkpoint issue by automatically discovering any available `.pt/.bin` checkpoints under `/kaggle/input` and using them if present; if none are available, the script still generate a valid submission (fallback rule) rather than erroring. Finally, I fix the offsets collection bug (offsets were empty because `__getitem__` is not guaranteed to be called in order) by returning offsets per-sample and carrying them through the collate/predict path so decoding is correct and stable, producing a properly formatted `submission.csv`.'
- What this solution (achieved 0.59324) has done: 'I fix the crash caused by passing `token_type_ids` into RoBERTa (which triggers a protobuf-related failure path in this environment) by making the forward pass omit `token_type_ids` when the underlying model doesn’t support it; this preserves the same architecture and outputs otherwise. I also fix a logic bug in `MyCollator` (it currently builds only 7 dummy tensors but `__getitem__` returns 8), which would break batching once cell 7 runs. Finally, I keep offsets aligned and ensure the submission is written exactly as required; no changes to training/inference semantics beyond the necessary compatibility fix.'
- What this solution (achieved 0.59324) has done: 'I fix the crash in model creation by switching from `AutoModel.from_pretrained` to a protobuf-free construction path (`AutoConfig.from_pretrained` + `AutoModel.from_config`) while still requesting `output_hidden_states=True`, which preserves the same core model architecture and forward semantics needed by your heads. I also ensure the token offset mappings are robustly handled by converting `None` offset entries to safe `(0,0)` pairs so decoding can’t fail silently or return empty spans unnecessarily. Finally, I keep checkpoint discovery/loading and the submission-writing logic intact so the notebook runs end-to-end and produces a valid `submission.csv`.'
- What this solution (achieved 0.59324) has done: 'I fix the crash happening when creating the model by avoiding the `AutoConfig.from_pretrained` path that triggers the protobuf `MessageFactory.GetPrototype` error in this environment, and instead load the backbone with `AutoModel.from_pretrained(..., output_hidden_states=True)` (same architecture/forward semantics). I also ensure the config has `output_hidden_states=True` even when a `config` is passed, and keep the token_type_ids omission for RoBERTa intact. These changes are minimal, unblock end-to-end execution, and allow your existing checkpoint-loading + ensembling logic to run, which should raise the score back toward your target by using the intended trained weights. The submission writing and format checks remain unchanged.'
- What this solution (achieved 0.59324) has done: 'I fix the crash in model creation (`MessageFactory.GetPrototype`) by avoiding the `AutoModel.from_pretrained` code path that triggers protobuf usage in this environment, and instead construct the backbone from a local pretrained config + weights while still enabling `output_hidden_states=True` (same architecture/forward semantics for your heads). I keep the rest of the pipeline intact (tokenization, heads, ensembling, decoding, submission format) and only adjust the minimal code needed to load the model safely. This should restore proper inference with any discovered checkpoints and move the score back upward toward your target, instead of falling back to the trivial full-text selection behavior. The script still produce a valid `submission.csv` even if no checkpoints are found.'
- What this solution (achieved 0.59324) has done: 'I fix the runtime crash in model construction by avoiding the `AutoConfig.from_pretrained` / protobuf-dependent path that triggers `MessageFactory.GetPrototype` in this environment, while keeping the same RoBERTa backbone, hidden-state usage, and heads. Concretely, I load the transformer via `AutoModel.from_pretrained(..., output_hidden_states=True)` and keep the existing forward logic (including omitting `token_type_ids` for RoBERTa). This is a bug fix that also restores the intended checkpoint-based inference (instead of falling back to full-text), which should move the score upward toward your target. I keep all data paths and submission writing unchanged, and ensure offsets remain correctly carried through batching for decoding.'
- What this solution (achieved 0.59324) has done: 'I fix the crash in model construction (`MessageFactory.GetPrototype`) by avoiding the protobuf-triggering `AutoModel.from_pretrained` code path for the backbone and instead loading weights with `AutoModel.from_config` + `load_state_dict` from the local cached `roberta-base` files. This keeps the same RoBERTa architecture and still enables `output_hidden_states=True`, so your downstream CNN/heads and decoding logic remain unchanged. I also keep the existing checkpoint discovery/loading logic intact so inference uses the intended trained weights (which should move the score up toward your target), while preserving the same submission format and offsets handling.'

# 9. Code solution

## === cell 0
import os
import re
import random
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from transformers import AutoConfig, AutoModel, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model: nn.Module, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)


def _clean_text(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def _find_best_span_from_probs(
    start_p: torch.Tensor, end_p: torch.Tensor, max_len: int = 30
) -> Tuple[int, int]:
    L = start_p.shape[0]
    best_score = -1.0
    best_i, best_j = 0, 0
    for i in range(L):
        j_max = min(L - 1, i + max_len)
        scores = start_p[i] * end_p[i : j_max + 1]
        j_rel = int(torch.argmax(scores).item())
        score = float(scores[j_rel].item())
        j = i + j_rel
        if score > best_score:
            best_score = score
            best_i, best_j = i, j
    return best_i, best_j


def _decode_selected_text_from_offsets(
    text: str, offsets: List[Tuple[int, int]], i: int, j: int
) -> str:
    if not offsets:
        return text.strip() if text else ""
    i = max(0, min(i, len(offsets) - 1))
    j = max(0, min(j, len(offsets) - 1))
    if j < i:
        i, j = j, i

    start_char = offsets[i][0]
    end_char = offsets[j][1]
    if start_char is None or end_char is None:
        return text.strip() if text else ""
    start_char = max(0, int(start_char))
    end_char = max(start_char, int(end_char))
    return text[start_char:end_char].strip() if text else ""


def ensemble(
    all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, df, softmax=False
):
    whole = np.mean(np.stack(all_whole_preds, axis=0), axis=0)

    n = len(df)
    start_list, end_list, inst_list = [], [], []
    for idx in range(n):
        s = torch.stack(
            [all_start_preds[f][idx] for f in range(len(all_start_preds))], dim=0
        ).mean(dim=0)
        e = torch.stack(
            [all_end_preds[f][idx] for f in range(len(all_end_preds))], dim=0
        ).mean(dim=0)
        inst = torch.stack(
            [all_inst_preds[f][idx] for f in range(len(all_inst_preds))], dim=0
        ).mean(dim=0)
        start_list.append(s)
        end_list.append(e)
        inst_list.append(inst)
    return whole, start_list, end_list, inst_list


def get_predicts_from_token_logits(
    all_whole_pred, all_start_pred, all_end_pred, all_inst_out, df, args
):
    word_preds = []
    inst_word_preds = []
    scores = []  # placeholder for compatibility

    for i in range(len(df)):
        text = _clean_text(df.loc[i, "text"])
        sentiment = _clean_text(df.loc[i, "sentiment"])

        if sentiment == "neutral":
            word_preds.append(text.strip())
            inst_word_preds.append(text.strip())
            scores.append(0.0)
            continue

        start_p = all_start_pred[i]
        end_p = all_end_pred[i]
        offsets = args._offsets[i] if hasattr(args, "_offsets") else None

        si, ei = _find_best_span_from_probs(start_p, end_p, max_len=30)
        sel = _decode_selected_text_from_offsets(text, offsets, si, ei)

        if sel == "":
            sel = text.strip()

        word_preds.append(sel)
        inst_word_preds.append(sel)
        scores.append(0.0)

    return word_preds, inst_word_preds, scores




## === cell 1
class TrainDataset(Dataset):
    """
    Minimal dataset for inference matching the original interface.
    Returns (tokens, types, masks, offsets, placeholders...) so predict() can decode correctly.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y=None,
        tokenizer=None,
        mode="test",
        offset=4,
        max_len=128,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = max_len
        self.offset = offset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        text = _clean_text(self.df.loc[idx, "text"])
        sentiment = _clean_text(self.df.loc[idx, "sentiment"])

        enc = self.tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            padding=False,
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attn = torch.tensor(enc["attention_mask"], dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)

        offsets = enc.get("offset_mapping", None)
        if offsets is None:
            offsets = []
        else:
            offsets = [
                (0, 0) if (o is None or o[0] is None or o[1] is None) else o
                for o in offsets
            ]

        dummy = torch.tensor(0, dtype=torch.long)
        return (
            input_ids,
            token_type_ids,
            attn,
            offsets,  # not a tensor; collator will keep as python list
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,  # 8 dummies total after offsets
        )


class MyCollator:
    def __call__(self, batch):
        input_ids = [b[0] for b in batch]
        token_type_ids = [b[1] for b in batch]
        attention_mask = [b[2] for b in batch]
        offsets = [b[3] for b in batch]
        rest = [b[4:] for b in batch]  # 8 dummies

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=1
        )  # RoBERTa pad token id = 1
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        dummies = []
        for k in range(8):
            dummies.append(torch.stack([r[k] for r in rest], dim=0))
        return (input_ids, token_type_ids, attention_mask, offsets, *dummies)




## === cell 2
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sample_sub = pd.read_csv(
    "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
)
assert list(sample_sub.columns) == ["textID", "selected_text"]



## === cell 3
MODEL_ID = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)


class Args:
    post = False
    tokenizer = None
    offset = 4
    batch_size = 16
    workers = 0  # safer in Kaggle notebook / docker


args = Args()
args.tokenizer = tokenizer



## === cell 4
collator = MyCollator()
test_set = TrainDataset(
    test, None, tokenizer=tokenizer, mode="test", offset=args.offset, max_len=128
)
test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collator,
    num_workers=args.workers,
    pin_memory=torch.cuda.is_available(),
)




## === cell 5
def _build_backbone_no_protobuf(pretrain_path: str):
    """
    Bugfix: avoid protobuf-triggering AutoModel.from_pretrained path in this environment.
    We still load the exact same roberta-base architecture and weights from local cache.
    """
    cfg = AutoConfig.from_pretrained(pretrain_path)
    cfg.output_hidden_states = True

    backbone = AutoModel.from_config(cfg)

    resolved = None
    state_dict = None

    try:
        tmp_cfg = AutoConfig.from_pretrained(pretrain_path, local_files_only=True)
        _ = tmp_cfg  # quiet lint
    except Exception:
        pass

    try:
        from transformers.utils import cached_file

        for fname in ("model.safetensors", "pytorch_model.bin"):
            try:
                resolved = cached_file(pretrain_path, fname, local_files_only=True)
            except Exception:
                resolved = None
            if resolved and os.path.exists(resolved):
                if fname.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    state_dict = load_file(resolved)
                else:
                    state_dict = torch.load(resolved, map_location="cpu")
                break
    except Exception:
        resolved = None
        state_dict = None

    if state_dict is None:
        backbone = AutoModel.from_pretrained(pretrain_path, output_hidden_states=True)
        return backbone

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    return backbone


class TweetModel(nn.Module):
    def __init__(self, pretrain_path=None, dropout=0.2, config=None):
        super(TweetModel, self).__init__()

        if config is None:
            self.bert = _build_backbone_no_protobuf(pretrain_path)
        else:
            try:
                config.output_hidden_states = True
            except Exception:
                pass
            self.bert = _build_backbone_no_protobuf(pretrain_path)

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
        model_type = getattr(getattr(self.bert, "config", None), "model_type", "")
        use_token_type = (token_type_ids is not None) and (
            model_type not in ("roberta", "xlm-roberta")
        )

        if use_token_type:
            out = self.bert(
                input_ids=inputs,
                attention_mask=masks,
                token_type_ids=token_type_ids,
                inputs_embeds=input_emb,
            )
        else:
            out = self.bert(
                input_ids=inputs,
                attention_mask=masks,
                inputs_embeds=input_emb,
            )

        hs = out.hidden_states  # tuple of layers
        seq_output = torch.cat([hs[-1], hs[-2], hs[-3]], dim=-1)

        avg_output = F.adaptive_avg_pool1d(seq_output.permute(0, 2, 1), 1).squeeze(-1)
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0, 2, 1)).permute(0, 2, 1))

        se_out = self.se_head(self.dropout(seq_output))
        inst_out = self.inst_head(self.dropout(seq_output))
        return whole_out, se_out[:, :, 0], se_out[:, :, 1], inst_out




## === cell 6
def predict(model: nn.Module, valid_df, valid_loader, args, progress=False):
    model.eval()
    all_end_pred, all_whole_pred, all_start_pred, all_inst_out = [], [], [], []
    all_offsets = []

    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for tokens, types, masks, offsets, _, _, _, _, _, _, _, _ in valid_loader:
            bs = tokens.size(0)
            if progress:
                tq.update(bs)

            all_offsets.extend(offsets)

            tokens = tokens.to(device)
            masks = masks.to(device)
            types = types.to(device)

            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)

            start_out = start_out.masked_fill(~masks.bool(), -1000)
            end_out = end_out.masked_fill(~masks.bool(), -1000)

            start_out = torch.softmax(start_out, dim=-1)
            end_out = torch.softmax(end_out, dim=-1)

            all_whole_pred.append(
                torch.softmax(whole_out, dim=-1)[:, 1].detach().cpu().numpy()
            )

            inst_out = torch.softmax(inst_out, dim=-1)
            for i in range(bs):
                all_start_pred.append(start_out[i, :].detach().cpu())
                all_end_pred.append(end_out[i, :].detach().cpu())
                all_inst_out.append(inst_out[i, :, 1].detach().cpu())

    all_whole_pred = np.concatenate(all_whole_pred, axis=0)
    if progress:
        tq.close()

    args._offsets = all_offsets
    return all_whole_pred, all_start_pred, all_end_pred, all_inst_out




## === cell 7
model = TweetModel(pretrain_path=MODEL_ID, config=None).to(device)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
def _discover_checkpoints(root="/kaggle/input"):
    ckpts = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            lfn = fn.lower()
            if lfn.endswith(".pt") or lfn.endswith(".bin"):
                ckpts.append(os.path.join(dirpath, fn))
    ckpts = sorted(ckpts)
    return ckpts


all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = [], [], [], []

CKPT_DIR = "/kaggle/input/roberta-weights-v10"
wanted = [os.path.join(CKPT_DIR, f"best-model-{fold}.pt") for fold in range(5)]
available = [p for p in wanted if os.path.exists(p)]

if len(available) == 0:
    discovered = _discover_checkpoints("/kaggle/input")
    available = discovered[:5]

if len(available) > 0:
    for ckpt_path in available:
        load_model(model, ckpt_path)
        model.to(device)
        fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = predict(
            model, test, test_loader, args, progress=True
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
        test,
        softmax=False,
    )

    word_preds, inst_word_preds, scores = get_predicts_from_token_logits(
        all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test, args
    )
else:
    word_preds = []
    for i in range(len(test)):
        text = _clean_text(test.loc[i, "text"])
        sentiment = _clean_text(test.loc[i, "sentiment"])
        if sentiment == "neutral":
            word_preds.append(text.strip())
        else:
            word_preds.append(text.strip())



## === cell 9
test["selected_text"] = word_preds

sub = test[["textID", "selected_text"]].copy()
sub["selected_text"] = sub["selected_text"].fillna("").astype(str)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 10
assert sub.shape[0] == sample_sub.shape[0], (sub.shape, sample_sub.shape)
assert (
    sub["textID"].values == sample_sub["textID"].values
).all(), "textID order mismatch vs sample submission"
sub.head(20)
