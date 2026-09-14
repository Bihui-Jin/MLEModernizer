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

0.59324

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I remove the dependency on missing `utils-v10` and `dataset10` files by inlining the small required functions/classes (dataset, collator, checkpoint loader, ensembling, and span-to-text decoding) so the notebook can run end-to-end. I also fix the HuggingFace path errors by loading `roberta-base` from the installed Transformers model hub with `local_files_only=True` fallback logic, instead of the non-existent `../input/roberta-base/` directory. Finally, I ensure inference produces valid `selected_text` strings aligned to `textID`, and always writes a correctly formatted `submission.csv` with the required header/quoting.'
- What this solution (achieved 0.59324) has done: 'The crash comes from an incompatibility between `transformers==4.53.3` importing protobuf internals and the runtime’s protobuf version, triggered when calling `RobertaConfig.from_pretrained`. To keep the exact model logic while fixing this, I avoid `RobertaConfig` entirely and load the model/config via `AutoConfig`/`AutoModel` with `output_hidden_states=True`, which bypasses the failing code path. I also fix a silent but important decoding bug: `predict()` currently applies `softmax` across the *sequence length* dimension (wrong), and then applies another `softmax`, which hurts span selection and score; changing it to apply softmax over the correct dimension (no extra softmax) preserves the intended semantics and should move the score toward your target. Everything else (architecture, checkpoints, ensembling, submission writing) remains the same.'
- What this solution (achieved 0.59324) has done: 'I fix the crash caused by a protobuf/transformers incompatibility that happens when calling `AutoConfig.from_pretrained` by avoiding that code path entirely and instead instantiating the backbone with `AutoModel.from_pretrained(..., output_hidden_states=True)` (this preserves the same architecture/heads and hidden-state usage). I also remove the incorrect span-probability computation: the model outputs per-token logits already, so we should apply `softmax` over the token dimension once (and not apply any extra/incorrect softmax), which should improve span selection and move the Jaccard score toward your target. I keep the rest of the pipeline (dataset/collator, ensembling, decoding, submission formatting) unchanged and ensure `submission.csv` is always written with the correct columns and row alignment. Finally, I keep the same paths and add robust local/offline loading fallbacks without changing training/inference semantics.'
- What this solution (achieved 0.59324) has done: 'I fix the runtime crash caused by the protobuf/transformers interaction that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` when importing/instantiating HuggingFace models. The smallest safe fix in a Kaggle/offline environment is to force Transformers to use the pure-Python protobuf implementation *before* importing `transformers`, which bypasses the broken C++ implementation path. I also make model loading robust by falling back to a tiny randomly initialized RoBERTa config if `roberta-base` files are not available locally (so the notebook always runs end-to-end and writes `submission.csv`). Core model architecture, inference logic, decoding, and submission formatting are otherwise preserved.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import random
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModel, AutoTokenizer, RobertaConfig, RobertaModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

        start_tok = best_j + args.offset
        end_tok = best_k + args.offset

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
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "roberta-base", use_fast=True, local_files_only=True
    )
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)




## === cell 4
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




## === cell 5
class TweetModel(nn.Module):
    def __init__(self, pretrain_path=None, dropout=0.2, backbone=None):
        super(TweetModel, self).__init__()
        if backbone is not None:
            self.bert = backbone
        else:
            self.bert = AutoModel.from_pretrained(
                pretrain_path, output_hidden_states=True
            )

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
        out = self.bert(
            inputs,
            attention_mask=masks,
            token_type_ids=token_type_ids,
            inputs_embeds=input_emb,
        )
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
def build_backbone():
    try:
        return AutoModel.from_pretrained(
            "roberta-base", output_hidden_states=True, local_files_only=True
        )
    except Exception:
        try:
            return AutoModel.from_pretrained("roberta-base", output_hidden_states=True)
        except Exception:
            cfg = RobertaConfig(
                vocab_size=50265,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=514,
                type_vocab_size=1,
                output_hidden_states=True,
            )
            return RobertaModel(cfg)


backbone = build_backbone()
model = TweetModel(backbone=backbone).to(DEVICE)



## === cell 8
CKPT_DIR_CANDIDATES = [
    "/kaggle/input/roberta-weights-v10",
    "/kaggle/data/roberta-weights-v10",
    "../input/roberta-weights-v10",
]
ckpt_dir = None
for c in CKPT_DIR_CANDIDATES:
    if os.path.isdir(c):
        ckpt_dir = c
        break

all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = [], [], [], []

if ckpt_dir is None:
    word_preds = test["text"].fillna("").astype(str).tolist()
else:
    for fold in range(5):
        ckpt_path = os.path.join(ckpt_dir, f"best-model-{fold}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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



## === cell 9
sub = pd.DataFrame({"textID": test["textID"], "selected_text": word_preds})
sub["selected_text"] = sub["selected_text"].fillna("").astype(str)

sub = sample_sub[["textID"]].merge(sub, on="textID", how="left")
sub["selected_text"] = sub["selected_text"].fillna("").astype(str)

sub.to_csv("submission.csv", index=False)
sub.head(10)



## === cell 10
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["textID", "selected_text"]
assert len(chk) == len(sample_sub) == len(test)
chk.isna().sum(), chk.head()
