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

0.690662145614624

# 6. Current score

0.586

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.586) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by making the Transformers import lazy and falling back to a compatible, locally-available model/tokenizer path (Kaggle’s `roberta-base`) when the custom `/kaggle/input/roberta/` files are missing. Then I ensure all required globals (SEED, BATCH_SIZE, etc.) are defined by unblocking cell execution order and removing reliance on variables from failed cells. Finally, I make inference robust by (a) loading fold checkpoints only if present and otherwise running a single model with base weights, and (b) always writing a correctly formatted `submission.csv` with the proper columns and quoting handled by pandas.'
- What this solution (achieved 0.586) has done: 'I fix the `MessageFactory.GetPrototype` crash by preventing `transformers` from importing protobuf-backed code paths, using the fast tokenizer without offset mappings and instead reconstructing token-to-text spans via `RobertaTokenizerFast.convert_ids_to_tokens` plus a deterministic character scan. This keeps the RoBERTa core model, training loop, loss, and inference semantics the same (start/end over token positions), but removes the failing dependency and restores correct offset behavior needed for good Jaccard. I also ensure the sentiment “special token” IDs match RoBERTa’s encoding (not hardcoded integers) to improve alignment and score toward your target. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.586) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by ensuring the protobuf environment variable is set before any `transformers` import, and by actively removing already-imported `google.protobuf` modules if they were loaded earlier in the notebook/session. I also make the RoBERTa model/tokenizer loading more robust by preferring Kaggle’s local `roberta-base` if present (but keeping the same architecture and start/end token objective). These changes are execution-stability focused but should also restore proper training/inference behavior (and therefore improve Jaccard versus the current broken state). The rest of the pipeline (dataset building, model layers, loss, inference averaging, and submission writing) is kept intact.'
- What this solution (achieved 0.586) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before any transformers-related import occurs* (and by removing any already-imported protobuf modules), which is the root cause of the runtime failure in cell 2. Then I keep the existing RoBERTa start/end span logic intact but switch back to using the fast tokenizer’s native `offset_mapping` (when available) instead of the fragile token-to-char reconstruction; this is a minimal change that typically improves span alignment and should move Jaccard upward toward your target. Finally, I make sure the script always reaches submission writing and outputs a valid `submission.csv` with the correct columns.'
- What this solution (achieved 0.586) has done: 'I fix the `MessageFactory.GetPrototype` crash by ensuring the protobuf implementation is set before any protobuf/transformers import happens, and by explicitly removing any preloaded `google.protobuf` modules. To keep the model/span logic intact but improve score toward your target, I switch model/tokenizer loading to `AutoModel`/`AutoTokenizer` (still RoBERTa) which avoids the fragile protobuf-backed paths and restores reliable fast-tokenizer `offset_mapping` needed for accurate span-to-text reconstruction. I also add a safe local fallback path for `roberta-base` under Kaggle inputs, without changing training/inference semantics. Finally, I keep the submission-writing logic the same and ensure the pipeline always reaches `submission.csv` creation.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold

import tokenizers

print("Torch:", torch.__version__)
print("Tokenizers:", tokenizers.__version__)
print("CUDA available:", torch.cuda.is_available())


## === cell 1
train_path = "/kaggle/input/tweet-sentiment-extraction/train.csv"
test_path = "/kaggle/input/tweet-sentiment-extraction/test.csv"
sample_sub_path = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

train_df = pd.read_csv(train_path)
train_df["text"] = train_df["text"].astype(str)
train_df["selected_text"] = train_df["selected_text"].astype(str)
train_df["sentiment"] = train_df["sentiment"].astype(str)


def seedall(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True




## === cell 2
MAX_LEN = 192
PATH = "/kaggle/input/roberta/"
ROBERTAFOLD = "/kaggle/input/robertafolds/"

EPOCHS = 3
BATCH_SIZE = 32
SEED = 42
DROPOUT = 0.1
LEARNING_RATE = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seedall(SEED)


def resolve_roberta_resources():
    """
    Use a local HF model dir if available; otherwise fall back to 'roberta-base'.
    """
    candidates = [
        "/kaggle/input/roberta-base",
        "/kaggle/input/tweet-sentiment-extraction/roberta-base",
        PATH,  # legacy/custom path if mounted
    ]
    for p in candidates:
        if os.path.isdir(p) and any(
            os.path.exists(os.path.join(p, f))
            for f in ["config.json", "tokenizer.json", "vocab.json"]
        ):
            return {"hf_name": p}
    return {"hf_name": "roberta-base"}


def load_roberta_and_tokenizer():
    """
    Bugfix: avoid protobuf 'MessageFactory.GetPrototype' crash by using Auto* APIs,
    while keeping the exact same core logic (RoBERTa encoder + start/end head).

    Score improvement: use fast tokenizer offset_mapping for accurate span alignment.
    """
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            del sys.modules[m]

    resources = resolve_roberta_resources()

    try:
        import transformers
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        print("Transformers:", transformers.__version__)
    except Exception as e:
        raise RuntimeError(
            "Failed to import transformers even after forcing pure-python protobuf. "
            "This environment should support transformers; verify package integrity."
        ) from e

    hf_name = resources["hf_name"]

    tok_fast = AutoTokenizer.from_pretrained(
        hf_name, use_fast=True, add_prefix_space=True
    )
    conf = AutoConfig.from_pretrained(hf_name, output_hidden_states=True)
    model = AutoModel.from_pretrained(hf_name, config=conf)

    if (
        tok_fast.cls_token_id is None
        or tok_fast.sep_token_id is None
        or tok_fast.pad_token_id is None
    ):
        raise RuntimeError(
            "Tokenizer is missing required special token ids (cls/sep/pad)."
        )

    class _TokWrapper:
        """
        Minimal interface:
          - encode(text).ids
          - encode(text).offsets (via offset_mapping)
        Also exposes special token ids and sentiment token ids.
        """

        def __init__(self, tok):
            self.tok = tok
            self.cls_id = tok.cls_token_id
            self.sep_id = tok.sep_token_id
            self.pad_id = tok.pad_token_id

        def encode(self, text):
            enc = self.tok(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_offsets_mapping=True,
            )
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]

            class _Enc:
                pass

            o = _Enc()
            o.ids = ids
            o.offsets = offsets
            return o

        def sentiment_ids(self, sentiment_text: str):
            return self.tok(
                sentiment_text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

    tokenizer = _TokWrapper(tok_fast)
    return tokenizer, conf, model


TOKENIZER, RobertaConf, MODEL = load_roberta_and_tokenizer()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.max_len = MAX_LEN
        self.labeled = "selected_text" in df.columns
        self.tokenizer = TOKENIZER

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet, offsets = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweet"] = tweet
        data["offsets"] = offsets

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx

        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        tweet = " " + " ".join(row.text.lower().split())
        encoding = self.tokenizer.encode(tweet)

        sentiment_id = self.tokenizer.sentiment_ids(row.sentiment.lower())

        cls_id = self.tokenizer.cls_id
        sep_id = self.tokenizer.sep_id
        pad_id = self.tokenizer.pad_id

        ids = [cls_id] + sentiment_id + [sep_id, sep_id] + encoding.ids + [sep_id]
        offsets = (
            [(0, 0)] * (1 + len(sentiment_id) + 2) + list(encoding.offsets) + [(0, 0)]
        )

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            offsets = offsets[: self.max_len]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [pad_id] * pad_len
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != pad_id).long()
        offsets = torch.tensor(offsets, dtype=torch.long)

        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(row.selected_text.lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        if len(selected_text) <= 1:
            return torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long)

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind : ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets.tolist()):
            if offset1 == 0 and offset2 == 0:
                continue
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        if len(target_idx) == 0:
            return torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]

        return torch.tensor(start_idx, dtype=torch.long), torch.tensor(
            end_idx, dtype=torch.long
        )


def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    train_part = df.iloc[train_idx].reset_index(drop=True)
    val_part = df.iloc[val_idx].reset_index(drop=True)

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_part),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_part),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return {"train": train_loader, "val": val_loader}


def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return loader




## === cell 4
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        config = RobertaConf
        self.roberta = MODEL
        self.dropout = nn.Dropout(DROPOUT)
        m_size = 128
        self.qa_outputs1c = torch.nn.Conv1d(config.hidden_size, m_size, 2)
        self.qa_outputs2c = torch.nn.Conv1d(config.hidden_size, m_size, 2)

        self.qa_outputs1 = nn.Linear(m_size, 1)
        self.qa_outputs2 = nn.Linear(m_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids, attention_mask)

        s_out = self.dropout(out[0])
        s_out = torch.nn.functional.pad(s_out.transpose(1, 2), (1, 0))

        out1 = self.qa_outputs1c(s_out).transpose(1, 2)
        out2 = self.qa_outputs2c(s_out).transpose(1, 2)

        start_logits = self.qa_outputs1(self.dropout(out1)).squeeze(-1)
        end_logits = self.qa_outputs2(self.dropout(out2)).squeeze(-1)
        return start_logits, end_logits




## === cell 5
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss




## === cell 6
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        o1, o2 = offsets[ix]
        if o1 == 0 and o2 == 0:
            continue
        selected_text += text[o1:o2]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)

    true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, pred)




## === cell 7
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    model.to(DEVICE)

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0

            for data in dataloaders_dict[phase]:
                ids = data["ids"].to(DEVICE)
                masks = data["masks"].to(DEVICE)
                tweet = data["tweet"]
                offsets = data["offsets"].cpu().numpy()
                start_idx = data["start_idx"].to(DEVICE)
                end_idx = data["end_idx"].to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    start_logits, end_logits = model(ids, masks)
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(ids)

                    start_idx_np = start_idx.detach().cpu().numpy()
                    end_idx_np = end_idx.detach().cpu().numpy()
                    start_probs = (
                        torch.softmax(start_logits, dim=1).detach().cpu().numpy()
                    )
                    end_probs = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

                    for i in range(len(ids)):
                        jaccard_score = compute_jaccard_score(
                            tweet[i],
                            start_idx_np[i],
                            end_idx_np[i],
                            start_probs[i],
                            end_probs[i],
                            offsets[i],
                        )
                        epoch_jaccard += jaccard_score

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

            print(
                "Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}".format(
                    epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard
                )
            )

    torch.save(model.state_dict(), filename)




## === cell 8
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

"""
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1):
    print(f"Fold: {fold}")

    model = TweetModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    criterion = loss_fn
    dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, BATCH_SIZE)

    train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        EPOCHS,
        f"roberta_fold{fold}.pth",
    )
"""


## === cell 9
test_df = pd.read_csv(test_path)
test_df["text"] = test_df["text"].astype(str)
test_df["sentiment"] = test_df["sentiment"].astype(str)

test_loader = get_test_loader(test_df, batch_size=BATCH_SIZE)

predictions = []
models = []

available_ckpts = []
for fold in range(skf.n_splits):
    ckpt_path = os.path.join(ROBERTAFOLD, f"roberta_fold{fold+1}.pth")
    if os.path.exists(ckpt_path):
        available_ckpts.append(ckpt_path)

if len(available_ckpts) == 0:
    model = TweetModel().to(DEVICE)
    model.eval()
    models = [model]
    print(
        "Warning: No fold checkpoints found in",
        ROBERTAFOLD,
        "-> using base model weights for inference.",
    )
else:
    for ckpt_path in available_ckpts:
        model = TweetModel().to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    print(f"Loaded {len(models)} checkpoint model(s).")

for data in test_loader:
    ids = data["ids"].to(DEVICE)
    masks = data["masks"].to(DEVICE)
    tweet = data["tweet"]
    offsets = data["offsets"].cpu().numpy()

    start_logits_folds = []
    end_logits_folds = []
    for model in models:
        with torch.no_grad():
            output = model(ids, masks)
            start_logits_folds.append(
                torch.softmax(output[0], dim=1).detach().cpu().numpy()
            )
            end_logits_folds.append(
                torch.softmax(output[1], dim=1).detach().cpu().numpy()
            )

    start_logits = np.mean(start_logits_folds, axis=0)
    end_logits = np.mean(end_logits_folds, axis=0)

    for i in range(len(ids)):
        start_pred = int(np.argmax(start_logits[i]))
        end_pred = int(np.argmax(end_logits[i]))
        if start_pred > end_pred:
            pred = tweet[i]
        else:
            pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
        predictions.append(pred)

if len(predictions) != len(test_df):
    raise RuntimeError(
        f"Prediction count mismatch: got {len(predictions)}, expected {len(test_df)}"
    )


## === cell 10
sub_df = pd.read_csv(sample_sub_path)

sub_df["selected_text"] = predictions

sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(str(x).split()) == 1 else x
)

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub_df.head())
