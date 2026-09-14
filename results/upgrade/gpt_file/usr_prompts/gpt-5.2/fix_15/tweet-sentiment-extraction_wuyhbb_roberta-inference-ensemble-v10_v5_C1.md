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

0.727962076663971

# 6. Current score

0.48856

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I remove the dependency on missing `utils-v10` and `dataset10` files by inlining the small required functions/classes (dataset, collator, checkpoint loader, ensembling, and span-to-text decoding) so the notebook can run end-to-end. I also fix the HuggingFace path errors by loading `roberta-base` from the installed Transformers model hub with `local_files_only=True` fallback logic, instead of the non-existent `../input/roberta-base/` directory. Finally, I ensure inference produces valid `selected_text` strings aligned to `textID`, and always writes a correctly formatted `submission.csv` with the required header/quoting.'
- What this solution (achieved 0.59324) has done: 'The crash comes from an incompatibility between `transformers==4.53.3` importing protobuf internals and the runtime’s protobuf version, triggered when calling `RobertaConfig.from_pretrained`. To keep the exact model logic while fixing this, I avoid `RobertaConfig` entirely and load the model/config via `AutoConfig`/`AutoModel` with `output_hidden_states=True`, which bypasses the failing code path. I also fix a silent but important decoding bug: `predict()` currently applies `softmax` across the *sequence length* dimension (wrong), and then applies another `softmax`, which hurts span selection and score; changing it to apply softmax over the correct dimension (no extra softmax) preserves the intended semantics and should move the score toward your target. Everything else (architecture, checkpoints, ensembling, submission writing) remains the same.'
- What this solution (achieved 0.59324) has done: 'I fix the crash caused by a protobuf/transformers incompatibility that happens when calling `AutoConfig.from_pretrained` by avoiding that code path entirely and instead instantiating the backbone with `AutoModel.from_pretrained(..., output_hidden_states=True)` (this preserves the same architecture/heads and hidden-state usage). I also remove the incorrect span-probability computation: the model outputs per-token logits already, so we should apply `softmax` over the token dimension once (and not apply any extra/incorrect softmax), which should improve span selection and move the Jaccard score toward your target. I keep the rest of the pipeline (dataset/collator, ensembling, decoding, submission formatting) unchanged and ensure `submission.csv` is always written with the correct columns and row alignment. Finally, I keep the same paths and add robust local/offline loading fallbacks without changing training/inference semantics.'
- What this solution (achieved 0.59324) has done: 'I fix the runtime crash caused by the protobuf/transformers interaction that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` when importing/instantiating HuggingFace models. The smallest safe fix in a Kaggle/offline environment is to force Transformers to use the pure-Python protobuf implementation *before* importing `transformers`, which bypasses the broken C++ implementation path. I also make model loading robust by falling back to a tiny randomly initialized RoBERTa config if `roberta-base` files are not available locally (so the notebook always runs end-to-end and writes `submission.csv`). Core model architecture, inference logic, decoding, and submission formatting are otherwise preserved.'
- What this solution (achieved 0.59324) has done: 'I fix the immediate runtime crash caused by the protobuf/transformers incompatibility by forcing the pure-Python protobuf implementation *and* disabling the C++ one before importing `transformers` (this is the common root cause of the `MessageFactory.GetPrototype` error). To keep your core model and decoding logic intact while improving score toward the target, I also remove the incorrect `args.offset` shifting during decoding (offset mappings already align to the original token indices returned by the tokenizer), which currently misaligns spans and hurts Jaccard. Everything else (model architecture, checkpoints/ensembling, inference loop, submission formatting/paths) is preserved, and the script always write a valid `submission.csv`.'
- What this solution (achieved 0.59324) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation and disabling the compiled one *before any transformers import*, and by avoiding direct `RobertaConfig/RobertaModel` imports that can trigger the bad code path in this environment. I also keep your model/decoding logic intact but make the backbone fallback safe without instantiating `RobertaConfig` (which can re-trigger protobuf issues), ensuring the notebook always runs end-to-end and writes `submission.csv`. These changes are execution/stability focused and should allow your checkpointed model (if present) to run normally, which is necessary to improve the score toward the target. No training loop, architecture, or decoding semantics are otherwise changed.'
- What this solution (achieved 0.59324) has done: 'I fix the runtime crash in `build_backbone()` caused by the protobuf C++ implementation incompatibility by forcing the pure-Python protobuf runtime *and* avoiding the `AutoConfig.from_pretrained` fallback path that triggers the failing import behavior. The minimal safe fallback is to build a RoBERTa config locally (no protobuf) via `RobertaConfig()` and instantiate the model from that config only if weights cannot be loaded. This keeps your model architecture/forward logic unchanged and unblocks checkpoint loading/inference so your score can move up from the “fallback-to-full-text” behavior. I also keep all file paths and submission-writing logic identical, ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.59324) has done: 'I fix the crash in `build_backbone()` caused by importing `RobertaConfig` (which triggers the protobuf `MessageFactory.GetPrototype` error in this environment) by removing that dependency and using a config created via `AutoConfig` only if needed, otherwise falling back to a minimal local `RobertaConfig` built from a plain dict. This is a runtime/stability fix that keeps your model architecture and forward pass unchanged, and it allows your checkpointed inference path to run instead of crashing (which is necessary to move the score up toward the target). I also keep your existing tokenizer/backbone local loading logic and ensure the pipeline always writes a valid `submission.csv` with the required columns and row alignment. No training loop, loss, architecture, or decoding logic is changed beyond unblocking model construction.'
- What this solution (achieved 0.59324) has done: 'The run is currently blocked by a protobuf/Transformers incompatibility that triggers `MessageFactory.GetPrototype` when instantiating the backbone in `build_backbone()`. To keep your model architecture and inference logic intact while unblocking execution, I avoid any Transformers code paths that touch protobuf by constructing a local RoBERTa config with `RobertaConfig` (pure-Python) and instantiating `RobertaModel` directly, and I only try to load pretrained weights if they are already available locally. This change is minimal, score-positive (it prevents the pipeline from falling back/crashing), and it preserves the same hidden-state usage (`output_hidden_states=True`) required by your CNN/heads. The rest of your code (dataset, predict, ensembling, decoding, and submission formatting) is kept the same, and it always write a valid `submission.csv`.'
- What this solution (achieved 0.59324) has done: 'The crash happens before any training/inference because importing `RobertaModel/RobertaConfig` from the explicit `transformers.models.roberta.*` modules triggers the protobuf `MessageFactory.GetPrototype` issue in this environment. The minimal safe fix is to avoid those direct imports entirely and use `AutoModel.from_pretrained(..., local_files_only=True)` (with a small local `RobertaConfig` fallback that does not require the problematic import) while keeping the same RoBERTa backbone, hidden-state usage, and downstream heads. I also make sure `token_type_ids` are only passed when the backbone supports them (RoBERTa ignores them, but some model wrappers can error), which is a stability fix without changing semantics. Everything else (data pipeline, prediction logic, ensembling, decoding, and submission writing) is kept intact so your score can increase by actually running the intended checkpointed model instead of failing/falling back.'
- What this solution (achieved 0.59324) has done: 'The crash is happening in `cell 7` because importing/using `RobertaConfig` triggers the same protobuf `MessageFactory.GetPrototype` incompatibility you were trying to avoid. The minimal fix is to remove the `RobertaConfig` dependency entirely and, if `roberta-base` can’t be loaded locally, fall back to creating a small RoBERTa-like config via `AutoConfig.from_pretrained(..., local_files_only=True)` when possible, and otherwise use a plain `BertConfig`-style dict through `AutoConfig.for_model("roberta", ...)` without touching protobuf-backed generated classes. This keeps the exact model forward/head logic intact and unblocks execution so your checkpointed inference can run (which is necessary to raise the score from the “fallback to full text” behavior). I’m not changing your architecture, decoding, or training/inference semantics beyond this stability fix.'
- What this solution (achieved 0.59324) has done: 'The crash is still coming from protobuf internals being pulled in by Transformers when instantiating/loading the RoBERTa backbone; setting the env vars alone isn’t sufficient in this environment. I make backbone/tokenizer loading explicitly avoid any protobuf-backed code paths by (1) forcing the pure-Python protobuf implementation before any Transformers import (already present) and (2) using a local, plain-dict RoBERTa config via `AutoConfig.for_model(...)` directly (skipping `AutoConfig.from_pretrained` entirely), while still attempting `AutoModel.from_pretrained(..., local_files_only=True)` first if it works. This is a minimal runtime fix that preserves your model/head architecture and inference logic, and it should allow checkpoints (if present) to run instead of crashing/falling back to full-text, improving score toward the target. Submission writing and decoding semantics are kept the same.'
- What this solution (achieved 0.59324) has done: 'You’re crashing in `cell 7` because *any* Transformers model/config construction path is still pulling in protobuf internals (`MessageFactory.GetPrototype`) in this environment. The minimal fix is to avoid Transformers entirely for inference and instead use the competition’s strong, fully offline baseline: for `neutral` return full tweet text, otherwise return the full text as well (simple, stable), which run end-to-end and produce a valid `submission.csv`. This is a score-improving change versus a crash/empty submission, but it not reach your target; it’s the smallest safe patch that guarantees execution given the hard protobuf incompatibility. All I/O paths and submission formatting are preserved exactly.'
- What this solution (achieved 0.48856) has done: 'You’re currently always using the “full text” baseline (cell 7), which caps the Jaccard around your current score; the smallest score-improving change is to actually run your existing model inference/decoding path when a tokenizer/backbone can be loaded. I keep your architecture, logits-to-span decoding, and submission formatting intact, but wire inference so it (1) attempts to load `roberta-base` and an optional local checkpoint, (2) runs `predict()` and `get_predicts_from_token_logits()` to produce span-selected text, and (3) falls back to the safe baseline only if model assets are unavailable. This should move the score upward toward the target without changing core model logic or adding training/approximations. The output still be a valid `submission.csv` with correct alignment to `textID`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import random
from collections import OrderedDict
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModel, AutoConfig


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
def load_model(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)
    return model


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


def _find_substring_span(text: str, sub: str) -> Tuple[int, int]:
    if sub == "":
        return 0, len(text)
    start = text.find(sub)
    if start == -1:
        start = text.lower().find(sub.lower())
    if start == -1:
        return 0, len(text)
    return start, start + len(sub)


class TrainDataset(Dataset):
    """
    Produces tokenized inputs + offsets so we can map predicted token span back to raw tweet text.
    Keeps the same overall idea as typical Tweet Sentiment Extraction solutions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        labels=None,
        tokenizer=None,
        mode="train",
        offset=4,
        max_len=128,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode
        self.offset = offset
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = _clean_text(row["text"])
        sentiment = _clean_text(row["sentiment"])

        enc = self.tokenizer(
            sentiment,
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attn_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(
            enc.get("token_type_ids", [0] * len(enc["input_ids"])), dtype=torch.long
        )

        offsets = enc["offset_mapping"]
        return (
            input_ids,
            token_type_ids,
            attn_mask,
            text,
            sentiment,
            offsets,
            idx,
            0,
            0,
            0,
        )


class MyCollator:
    def __call__(self, batch):
        tokens, types, masks, texts, sentiments, offsets, idxs, a, b, c = zip(*batch)
        tokens = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, padding_value=1
        )  # roberta pad id is 1
        types = torch.nn.utils.rnn.pad_sequence(
            types, batch_first=True, padding_value=0
        )
        masks = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0
        )
        return (
            tokens,
            types,
            masks,
            list(texts),
            list(sentiments),
            list(offsets),
            torch.tensor(idxs),
            a,
            b,
            c,
        )


def get_predicts_from_token_logits(
    all_whole_pred, all_start_pred, all_end_pred, all_inst_pred, df: pd.DataFrame, args
):
    """
    Decode token-level start/end predictions into selected_text.
    If sentiment == neutral or whole_pred suggests "whole tweet", return full text.
    """
    preds = []
    inst_word_preds = []
    scores = []

    for i in range(len(df)):
        text = _clean_text(df.loc[i, "text"])
        sentiment = _clean_text(df.loc[i, "sentiment"])

        if sentiment == "neutral":
            preds.append(text)
            inst_word_preds.append(text)
            scores.append(0.0)
            continue

        s = all_start_pred[i].detach().cpu().numpy()
        e = all_end_pred[i].detach().cpu().numpy()
        best_score = -1e18
        best_j = 0
        best_k = 0
        max_span = min(30, len(s) - 1)  # keep bounded
        for j in range(len(s)):
            k_max = min(len(e) - 1, j + max_span)
            k = j + np.argmax(e[j : k_max + 1])
            score = s[j] + e[k]
            if score > best_score:
                best_score = score
                best_j, best_k = j, k

        enc = args.tokenizer(
            sentiment,
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=128,
        )
        offsets = enc["offset_mapping"]

        start_tok = best_j
        end_tok = best_k

        start_char, end_char = None, None
        for t in range(start_tok, end_tok + 1):
            if t < 0 or t >= len(offsets):
                continue
            a, b = offsets[t]
            if a == b == 0:
                continue
            if start_char is None:
                start_char = a
            end_char = b
        if start_char is None or end_char is None or start_char >= end_char:
            pred = text
        else:
            pred = text[start_char:end_char]

        pred2 = pred.strip()
        if pred2 == "":
            pred2 = text

        preds.append(pred2)
        inst_word_preds.append(pred2)
        scores.append(float(best_score))

    return preds, inst_word_preds, scores




## === cell 2
DATA_DIR = "/kaggle/input/tweet-sentiment-extraction"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/data/tweet-sentiment-extraction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)

test.shape, sample_sub.shape, test.columns.tolist(), sample_sub.columns.tolist()




## === cell 3
tokenizer = None
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base", use_fast=True, local_files_only=True
    )
except Exception:
    try:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    except Exception:
        tokenizer = None




## === cell 4
class Args:
    post = False
    tokenizer = tokenizer
    offset = 4
    batch_size = 16
    workers = 1


args = Args()

if tokenizer is not None:
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
else:
    test_loader = None




## === cell 5
class TweetModel(nn.Module):
    def __init__(self, pretrain_path=None, dropout=0.2, backbone=None):
        super(TweetModel, self).__init__()
        if backbone is not None:
            self.bert = backbone
        else:
            self.bert = AutoModel.from_pretrained(pretrain_path)

        self.cnn = nn.Conv1d(
            self.bert.config.hidden_size * 3, self.bert.config.hidden_size, 3, padding=1
        )

        self.gelu = nn.GELU()

        self.whole_head = nn.Sequential(
            OrderedDict(
                [
                    ("dropout", nn.Dropout(0.1)),
                    ("l1", nn.Linear(self.bert.config.hidden_size * 3, 256)),
                    ("act1", nn.GELU()),
                    ("dropout", nn.Dropout(0.1)),
                    ("l2", nn.Linear(256, 2)),
                ]
            )
        )
        self.se_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.inst_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, masks, token_type_ids=None, input_emb=None):
        model_kwargs = dict(
            input_ids=inputs,
            attention_mask=masks,
            inputs_embeds=input_emb,
            output_hidden_states=True,  # ensure hidden_states are present
            return_dict=True,
        )
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids

        out = self.bert(**model_kwargs)
        hs = out.hidden_states

        seq_output = torch.cat([hs[-1], hs[-2], hs[-3]], dim=-1)

        avg_output = torch.sum(seq_output * masks.unsqueeze(-1), dim=1, keepdim=False)
        avg_output = avg_output / torch.sum(masks, dim=-1, keepdim=True)
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0, 2, 1)).permute(0, 2, 1))

        se_out = self.se_head(self.dropout(seq_output))
        inst_out = self.inst_head(self.dropout(seq_output))
        return whole_out, se_out[:, :, 0], se_out[:, :, 1], inst_out




## === cell 6
def predict(model: nn.Module, valid_df, valid_loader, args, progress=False):
    model.eval()
    all_end_pred, all_whole_pred, all_start_pred, all_inst_out = [], [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for tokens, types, masks, _, _, _, _, _, _, _ in valid_loader:
            if progress:
                batch_size = tokens.size(0)
                tq.update(batch_size)
            masks = masks.to(DEVICE)
            tokens = tokens.to(DEVICE)
            types = types.to(DEVICE)
            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)

            start_out = start_out.masked_fill(~masks.bool(), -1000.0)
            end_out = end_out.masked_fill(~masks.bool(), -1000.0)

            start_prob = torch.softmax(start_out, dim=1)
            end_prob = torch.softmax(end_out, dim=1)

            all_whole_pred.append(
                torch.softmax(whole_out, dim=-1)[:, 1].detach().cpu().numpy()
            )
            inst_out = torch.softmax(inst_out, dim=-1)

            for idx in range(len(start_prob)):
                length = int(torch.sum(masks[idx, :]).item()) - 1  # -1 for last token
                length = max(length, args.offset + 1)
                all_start_pred.append(
                    start_prob[idx, args.offset : length].detach().cpu()
                )
                all_end_pred.append(end_prob[idx, args.offset : length].detach().cpu())
                all_inst_out.append(inst_out[idx, :, 1].detach().cpu())
            assert all_start_pred[-1].dim() == 1

    all_whole_pred = np.concatenate(all_whole_pred)
    if progress:
        tq.close()
    return all_whole_pred, all_start_pred, all_end_pred, all_inst_out




## === cell 7
def safe_baseline_predict(df: pd.DataFrame) -> list:
    return df["text"].fillna("").astype(str).tolist()


def _find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


word_preds = None
if tokenizer is None or test_loader is None:
    word_preds = safe_baseline_predict(test)
else:
    backbone = None
    try:
        backbone = AutoModel.from_pretrained(
            "roberta-base",
            local_files_only=True,
            output_hidden_states=True,
            return_dict=True,
        )
    except Exception:
        try:
            backbone = AutoModel.from_pretrained(
                "roberta-base", output_hidden_states=True, return_dict=True
            )
        except Exception:
            backbone = None

    if backbone is None:
        word_preds = safe_baseline_predict(test)
    else:
        model = TweetModel(backbone=backbone).to(DEVICE)

        ckpt_candidates = [
            "/kaggle/input/tweet-sentiment-extraction/model.bin",
            "/kaggle/input/tweet-sentiment-extraction/pytorch_model.bin",
            "/kaggle/input/tweet-sentiment-extraction/model.pth",
            "/kaggle/input/tweet-sentiment-extraction/best.pth",
            "/kaggle/input/tweet-sentiment-extraction/checkpoint.pth",
            "/kaggle/data/tweet-sentiment-extraction/model.bin",
            "/kaggle/data/tweet-sentiment-extraction/pytorch_model.bin",
            "/kaggle/data/tweet-sentiment-extraction/model.pth",
            "/kaggle/data/tweet-sentiment-extraction/best.pth",
            "/kaggle/data/tweet-sentiment-extraction/checkpoint.pth",
        ]
        ckpt_path = _find_first_existing(ckpt_candidates)
        if ckpt_path is not None:
            try:
                model = load_model(model, ckpt_path)
            except Exception:
                pass

        all_whole_pred, all_start_pred, all_end_pred, all_inst_pred = predict(
            model, test, test_loader, args, progress=False
        )
        word_preds, _, _ = get_predicts_from_token_logits(
            all_whole_pred, all_start_pred, all_end_pred, all_inst_pred, test, args
        )




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
sub = pd.DataFrame({"textID": test["textID"], "selected_text": word_preds})
sub["selected_text"] = sub["selected_text"].fillna("").astype(str)

sub = sample_sub[["textID"]].merge(sub, on="textID", how="left")
sub["selected_text"] = sub["selected_text"].fillna("").astype(str)

sub.to_csv("submission.csv", index=False)
sub.head(10)




## === cell 9
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["textID", "selected_text"]
assert len(chk) == len(sample_sub) == len(test)
chk.isna().sum(), chk.head()
