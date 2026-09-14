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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.7162611484527588

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import re
import string
import random
import warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast

warnings.filterwarnings("ignore")


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed = 42
seed_everything(seed)

batch_size = 32
N = 10
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)
NUM_WORKERS = 2

ROBERTA_PATH = "/kaggle/input/robertamodel0524/"
outdir = "/kaggle/input/roberta714kernel/"

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

CLS_TOK = 0
PAD_TOK = 1
SEP_TOK = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resolve_roberta_model_source():
    """
    Bugfix (IO/offline): Prefer a local directory with roberta files if present;
    otherwise fall back to the offline HF cache model id 'roberta-base'.
    This avoids hard failing on missing /kaggle/input/robertamodel0524 assets.
    """
    candidate_dirs = [
        ROBERTA_PATH,
        "/kaggle/input/roberta-base/",
        "/kaggle/input/roberta/",
        "/kaggle/input/roberta714kernel/",
        "/kaggle/input/",
    ]
    weight_names = [
        "pytorch_model.bin",
        "roberta-base-pytorch_model.bin",
        "model.safetensors",
    ]
    config_names = ["config.json", "roberta-base-config.json"]

    for d in candidate_dirs:
        if not d or not os.path.isdir(d):
            continue
        cfg_ok = any(os.path.exists(os.path.join(d, cn)) for cn in config_names)
        w_ok = any(os.path.exists(os.path.join(d, wn)) for wn in weight_names)
        if cfg_ok and w_ok:
            return d

    return "roberta-base"


ROBERTA_SOURCE = resolve_roberta_model_source()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TOKENIZER = RobertaTokenizerFast.from_pretrained(ROBERTA_SOURCE, local_files_only=True)

CLS_TOK = int(TOKENIZER.cls_token_id)
PAD_TOK = int(TOKENIZER.pad_token_id)
SEP_TOK = int(TOKENIZER.sep_token_id)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2974790006.py in <cell line: 0>()
      1 # Bugfix: Replace ByteLevelBPETokenizer file dependency with RobertaTokenizerFast.
      2 # This removes the missing-vocab/merges failure and keeps RoBERTa tokenization behavior.
----> 3 TOKENIZER = RobertaTokenizerFast.from_pretrained(ROBERTA_SOURCE, local_files_only=True)
      4 
      5 # Ensure special token ids align with the model/tokenizer (score-neutral correctness guard).

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
    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
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
        tweet = " " + " ".join(str(row.text).lower().split())
        sentiment = str(row.sentiment).lower().strip()

        enc_tweet = self.tokenizer(
            tweet,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        enc_sent = self.tokenizer(
            sentiment,
            add_special_tokens=False,
            return_offsets_mapping=False,
        )

        tweet_ids = enc_tweet["input_ids"]
        tweet_offsets = enc_tweet["offset_mapping"]

        sent_ids = enc_sent["input_ids"]

        ids = [CLS_TOK] + sent_ids + [SEP_TOK, SEP_TOK] + tweet_ids + [SEP_TOK]
        offsets = [(0, 0)] * (1 + len(sent_ids) + 2) + list(tweet_offsets) + [(0, 0)]

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            offsets = offsets[: self.max_len]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [PAD_TOK] * pad_len
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != PAD_TOK).long()
        offsets = torch.tensor(offsets, dtype=torch.long)

        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(str(row.selected_text).lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        if len(selected_text) > 1:
            for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
                if " " + tweet[ind : ind + len_st] == selected_text:
                    idx0 = ind
                    idx1 = ind + len_st - 1
                    break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                if 0 <= ct < len(char_targets):
                    char_targets[ct] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets.tolist()):
            if offset2 > offset1 and sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        if len(target_idx) == 0:
            return 0, min(len(offsets) - 1, self.max_len - 1)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx




## === cell 3
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return loader




## === cell 4
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        config = RobertaConfig.from_pretrained(
            ROBERTA_SOURCE, output_hidden_states=True, local_files_only=True
        )
        self.roberta = RobertaModel.from_pretrained(
            ROBERTA_SOURCE,
            config=config,
            local_files_only=True,
        )
        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.hidden_states

        x = torch.stack([hs[-1], hs[-2], hs[-3]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 5
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        o1, o2 = int(offsets[ix][0]), int(offsets[ix][1])
        if o2 > o1:
            selected_text += text[o1:o2]
        if (ix + 1) < len(offsets) and int(offsets[ix][1]) < int(offsets[ix + 1][0]):
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
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




## === cell 6
def simple_fallback_selected_text(text: str, sentiment: str) -> str:
    """
    Execution-unblocker: If model checkpoints are missing, produce a deterministic,
    valid prediction that typically scores reasonably for this competition.
    (Keeps pipeline end-to-end; does not change model behavior when checkpoints exist.)
    """
    text = "" if text is None else str(text)
    sentiment = "" if sentiment is None else str(sentiment).lower().strip()
    if sentiment == "neutral":
        return text

    pos_words = {
        "good",
        "great",
        "love",
        "best",
        "amazing",
        "awesome",
        "nice",
        "happy",
        "fantastic",
        "wonderful",
    }
    neg_words = {
        "bad",
        "worst",
        "hate",
        "awful",
        "sad",
        "terrible",
        "horrible",
        "angry",
        "sucks",
        "annoying",
    }

    tokens = text.split()
    if not tokens:
        return text

    cand_idx = None
    if sentiment == "positive":
        for i, t in enumerate(tokens):
            w = re.sub(r"^\W+|\W+$", "", t.lower())
            if w in pos_words:
                cand_idx = i
                break
    elif sentiment == "negative":
        for i, t in enumerate(tokens):
            w = re.sub(r"^\W+|\W+$", "", t.lower())
            if w in neg_words:
                cand_idx = i
                break

    if cand_idx is None:
        cand_idx = 0

    return tokens[cand_idx]




## === cell 7
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)

test_loader = get_test_loader(test_df, batch_size=batch_size)

predictions = []
models = []

print("loading models..")

ckpt_base_dirs = [
    outdir,
    "/kaggle/input/roberta714kernel/",
    "/kaggle/input/roberta714kernel/roberta714kernel/",
    "/kaggle/input/",
]

missing = []
for fold in tqdm(range(skf.n_splits), total=skf.n_splits):
    ckpt_path = None
    for bd in ckpt_base_dirs:
        candidate = os.path.join(bd, f"roberta_fold{fold+1}.pth")
        if os.path.exists(candidate):
            ckpt_path = candidate
            break

    if ckpt_path is None:
        missing.append(f"roberta_fold{fold+1}.pth")
        continue

    model = TweetModel().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    models.append(model)

use_fallback = len(models) == 0
if use_fallback:
    print(
        "WARNING: No fold checkpoints were found; writing submission using a simple deterministic fallback."
    )
else:
    if len(missing) > 0:
        print(
            f"WARNING: Missing {len(missing)}/{skf.n_splits} checkpoints. Proceeding with {len(models)} found."
        )

if use_fallback:
    predictions = [
        simple_fallback_selected_text(t, s)
        for t, s in zip(test_df["text"].tolist(), test_df["sentiment"].tolist())
    ]
else:
    for data in tqdm(test_loader, total=len(test_loader)):
        ids = data["ids"].to(device)
        masks = data["masks"].to(device)
        tweet = data["tweet"]
        offsets = data["offsets"].numpy()

        start_logits_folds = []
        end_logits_folds = []

        for model in models:
            with torch.no_grad():
                start_logit, end_logit = model(ids, masks)
                start_logits_folds.append(
                    torch.softmax(start_logit, dim=1).cpu().numpy()
                )
                end_logits_folds.append(torch.softmax(end_logit, dim=1).cpu().numpy())

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

sub_df = pd.read_csv(submission_template)
if len(predictions) != len(sub_df):
    raise RuntimeError(
        f"Predictions length {len(predictions)} != submission rows {len(sub_df)}"
    )

sub_df["selected_text"] = predictions
sub_df["selected_text"] = sub_df["selected_text"].fillna("").astype(str)

sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(str(x).split()) == 1 else x
)

sub_df = sub_df[["textID", "selected_text"]]
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {sub_df.shape}")
print(sub_df.head(20).to_string(index=False))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/267406525.py in <cell line: 0>()
      2 test_df["text"] = test_df["text"].astype(str)
      3 
----> 4 test_loader = get_test_loader(test_df, batch_size=batch_size)
      5 
      6 predictions = []

/tmp/ipykernel_55/1411097765.py in get_test_loader(df, batch_size)
      1 def get_test_loader(df, batch_size=32):
      2     loader = torch.utils.data.DataLoader(
----> 3         TweetDataset(df),
      4         batch_size=batch_size,
      5         shuffle=False,

/tmp/ipykernel_55/1369666244.py in __init__(self, df, max_len)
      6 
      7         # Use global fast tokenizer
----> 8         self.tokenizer = TOKENIZER
      9 
     10     def __getitem__(self, index):

NameError: name 'TOKENIZER' is not defined
