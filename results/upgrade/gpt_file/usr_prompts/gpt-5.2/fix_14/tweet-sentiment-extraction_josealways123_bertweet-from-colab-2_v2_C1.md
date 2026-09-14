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

emoji==2.15.0
geopandas==0.14.4
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

0.7090124487876892

# 6. Current score

0.57635

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fix the two blockers that prevent any submission: (1) the protobuf/transformers import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before importing transformers, and (2) the missing RoBERTa weights/tokenizer in this offline environment by falling back to a lightweight, local, deterministic baseline that still produces a valid `submission.csv`. The fallback keeps the pipeline structure intact (load data → build dataset/loader → predict → write CSV) while avoiding any internet/model download dependency. This run end-to-end within the time limit and generate a correctly formatted submission file with quoted text handled by `to_csv`. If RoBERTa assets are actually present in your environment, the code automatically use them; otherwise it safely degrade to the baseline.'
- What this solution (achieved 0.57725) has done: 'I fix the transformers/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling C++ protobuf at process start, and I guard all transformer-dependent code so it can’t execute when transformers fails to import. Then, to move your score up toward the 0.709 target without changing the model/training core, I replace the very weak baseline (returning full tweet) with a simple sentiment-aware heuristic that selects a more plausible span (and returns full tweet for neutral), which is still deterministic and offline-safe. This should substantially improve the Jaccard score while keeping the rest of the pipeline intact and still producing a valid `submission.csv`. If local RoBERTa assets exist and transformers loads successfully, the original model path remains unchanged.'
- What this solution (achieved 0.57725) has done: 'I fix the transformers/protobuf import crash by forcing the pure-Python protobuf runtime *and* proactively disabling the C++ implementation via `google.protobuf.internal.api_implementation`, which is the common source of the `MessageFactory.GetPrototype` error in Kaggle images. This is a correctness/stability fix that allows the RoBERTa path (your actual model) to run instead of falling back to the heuristic baseline, which should move your score up toward the 0.709 target. I also make the DataLoader deterministic/safer in notebooks by setting `num_workers=0` (avoids multiprocessing import side-effects that can re-trigger protobuf/transformers issues) without changing the model/training logic. Submission writing and format are kept identical, still producing `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.57725) has done: 'I fix the `transformers/protobuf` crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation earlier and more robustly (including disabling the upb C++ backend before any protobuf/transformers import). This should allow the existing RoBERTa path to run (using local_files_only) instead of falling back to the heuristic, which is the minimal change most likely to raise the score toward the 0.709 target without altering your model/training logic. I also add a safe fallback so that if transformers still can’t load locally, the script continues to produce a valid `submission.csv`. Finally, I keep DataLoader workers at 0 to avoid subprocess re-import side effects that can re-trigger the protobuf issue in Kaggle.'
- What this solution (achieved 0.57725) has done: 'I fix the immediate crash in the transformers import path by force-disabling the C++/upb protobuf backend *before* any protobuf-dependent imports, which is the root cause of the `MessageFactory.GetPrototype` AttributeError in this environment. This should allow the existing RoBERTa inference path (and any provided fold checkpoints) to run instead of always falling back to the heuristic baseline, which is the smallest change likely to move your score up toward the 0.709 target. If transformers still cannot load locally, the code continue to safely fall back to the heuristic and still write a valid `submission.csv`. No model architecture/training loop logic is changed.'
- What this solution (achieved 0.57725) has done: 'I fix the `transformers/protobuf` crash by forcing the pure-Python protobuf implementation even earlier and more aggressively (including disabling the upb backend before `transformers` touches protobuf), so the RoBERTa path can actually run instead of always falling back to the heuristic. I also make the transformer-loading block robust: if import still fails, it cleanly fall back to the existing heuristic and still produce `submission.csv`. This is the smallest change that should move your score up toward the 0.709 target because it enables your original RoBERTa inference/training logic without changing the model, loss, or postprocessing. Finally, I keep `num_workers=0` to avoid subprocess re-import side effects that can re-trigger the protobuf issue.'
- What this solution (achieved 0.57725) has done: 'I fix the crash in the transformers import path (`MessageFactory.GetPrototype`) by forcing protobuf’s pure-Python runtime in a way that works with the current `protobuf` versions, and by preventing the C++/upb backend from being used before `transformers` is imported. This should allow the original RoBERTa inference/training codepath to execute (instead of always falling back to the heuristic), which is the smallest legitimate change likely to increase your score toward the 0.709 target. I also make the transformers-loading block more robust (catch both import and load failures cleanly) while keeping the same model, dataset, loss, and decoding logic. The script still always produce `/kaggle/working/submission.csv` even if transformers cannot be used.'
- What this solution (achieved 0.57725) has done: 'I fix the transformers/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation even earlier and more decisively (including setting env vars and attempting to switch the protobuf runtime before importing transformers). This should allow the existing RoBERTa inference path (and any provided fold checkpoints) to run instead of always falling back to the heuristic baseline, which is the smallest legitimate change likely to raise your score toward the 0.709 target. I also make the transformers import/load block robust so that if it still fails, the script cleanly falls back and still writes a valid `/kaggle/working/submission.csv`. No model architecture, loss, or decoding logic is changed.'
- What this solution (achieved 0.57725) has done: 'I fix the `transformers` import crash (`MessageFactory.GetPrototype`) by forcing protobuf’s pure-Python backend *before any protobuf/transformers import* and by setting the internal protobuf implementation type early, which unblocks the RoBERTa path instead of always falling back to the heuristic. I also make the local model directory detection stricter so it finds an actual RoBERTa folder (not just any Hugging Face cache folder), increasing the chance that `local_files_only=True` succeeds offline. If transformers still can’t be used, the existing heuristic fallback remains unchanged so you always get a valid `submission.csv`. These changes are execution/stability-focused and should legitimately move score upward toward the 0.709 target by enabling the original model inference.'
- What this solution (achieved 0.57635) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing protobuf’s pure-Python backend even earlier and more decisively, before any indirect protobuf usage can happen, and I wrap the transformers import in a broader guard so it cleanly falls back if anything still fails. Then, to move your score up toward the 0.709 target (from 0.577) without changing the RoBERTa model path, I improve only the offline heuristic fallback (used when transformers can’t load locally) by selecting a best-scoring contiguous token span using sentiment lexicons plus punctuation/elongation cues, instead of returning a single token/full tweet too often. I also ensure the submission is always aligned to `sample_submission.csv` order and that `predictions` length matches `test.csv`. These changes keep the existing training/inference logic intact when RoBERTa is available, but substantially improve the baseline when it isn’t.'

# 9. Code solution

## === cell 0
import os
import re
import random
import warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import sys

    if "google.protobuf.pyext._message" in sys.modules:
        del sys.modules["google.protobuf.pyext._message"]
except Exception:
    pass

try:
    import google.protobuf.internal.api_implementation as _api_impl  # noqa: F401

    try:
        _api_impl._SetType("python")
    except Exception:
        pass
    try:
        if hasattr(_api_impl, "_implementation_type"):
            _api_impl._implementation_type = "python"
    except Exception:
        pass
except Exception:
    pass

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

seed = 18
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

DATA_DIR = "/kaggle/input/tweet-sentiment-extraction"
WORK_DIR = "/kaggle/working"

os.makedirs(WORK_DIR, exist_ok=True)



## === cell 1
from nltk.tokenize import TweetTokenizer
from emoji import demojize

tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

    return " ".join(normTweet.split())




## === cell 2
normalizeTweet(" I`d have responded, if I were going")



## === cell 3
USE_TRANSFORMERS = False
MODEL_DIR = None
config = None
hf_tokenizer = None

RobertaModel = None
RobertaConfig = None
RobertaTokenizerFast = None


def _looks_like_roberta_dir(p: str) -> bool:
    if not p or not os.path.isdir(p):
        return False
    files = set(os.listdir(p))
    has_cfg = "config.json" in files
    has_vocab_merges = ("vocab.json" in files and "merges.txt" in files) or (
        "tokenizer.json" in files
    )
    return has_cfg and has_vocab_merges


def _find_any_local_roberta_dir():
    candidates = [
        "/kaggle/input/roberta-base",
        "/kaggle/input/bert-roberta/roberta-base",
        "/kaggle/input/transformers/roberta-base",
        "/kaggle/input/huggingface-roberta/roberta-base",
        "/kaggle/input/roberta-base-squad2",
        "/kaggle/input/roberta-base-pytorch/roberta-base",
        "/kaggle/input/cardiffnlp-twitter-roberta-base-sentiment",
        "/kaggle/input/twitter-roberta-base-sentiment",
        "/kaggle/input/roberta-base/roberta-base",
        "/kaggle/input/cardiffnlp-twitter-roberta-base-sentiment/cardiffnlp-twitter-roberta-base-sentiment",
    ]

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    candidates += [
        os.path.join(hf_home, "hub"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface/transformers"),
    ]

    for p in candidates:
        if _looks_like_roberta_dir(p):
            return p

    for base in [os.path.expanduser("~/.cache/huggingface/hub"), hf_home]:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            files = set(files)
            if "config.json" in files and (
                ("vocab.json" in files and "merges.txt" in files)
                or ("tokenizer.json" in files)
            ):
                return root
    return None


try:
    from transformers import RobertaModel as _RobertaModel
    from transformers import RobertaConfig as _RobertaConfig
    from transformers import RobertaTokenizerFast as _RobertaTokenizerFast

    RobertaModel = _RobertaModel
    RobertaConfig = _RobertaConfig
    RobertaTokenizerFast = _RobertaTokenizerFast

    MODEL_DIR = _find_any_local_roberta_dir()
    MODEL_ID_TRIES = [
        MODEL_DIR,
        "roberta-base",
        "cardiffnlp/twitter-roberta-base-sentiment",
    ]

    last_err = None
    for model_id in MODEL_ID_TRIES:
        if not model_id:
            continue
        try:
            config = RobertaConfig.from_pretrained(
                model_id, output_hidden_states=True, local_files_only=True
            )
            hf_tokenizer = RobertaTokenizerFast.from_pretrained(
                model_id, local_files_only=True
            )
            MODEL_DIR = model_id
            USE_TRANSFORMERS = True
            break
        except Exception as e:
            last_err = e
            continue

    if USE_TRANSFORMERS:
        print("Using local RoBERTa model/tokenizer from:", MODEL_DIR)
    else:
        print("No local RoBERTa assets found; falling back to heuristic baseline.")
        print("Last transformers loading error:", repr(last_err))
except Exception as e:
    USE_TRANSFORMERS = False
    print("Transformers import/load failed; falling back to heuristic baseline.")
    print("Error:", repr(e))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
if USE_TRANSFORMERS:

    class _HFLikeBPE:
        def __init__(self, tok):
            self.tok = tok

        def encode(self, text: str) -> str:
            toks = self.tok.tokenize(text, add_prefix_space=False)
            return " ".join(toks)

    class _HFLikeVocab:
        def __init__(self, tok):
            self.tok = tok

        def encode_line(
            self, token_str: str, append_eos=False, add_if_not_exist=False
        ) -> torch.LongTensor:
            toks = token_str.split() if isinstance(token_str, str) else list(token_str)
            ids = self.tok.convert_tokens_to_ids(toks)
            return torch.tensor(ids, dtype=torch.long)

    bpe = _HFLikeBPE(hf_tokenizer)
    vocab = _HFLikeVocab(hf_tokenizer)
else:
    bpe = None
    vocab = None




## === cell 5
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, bpe, vocab, max_len=96):
        self.df = df
        self.labeled = "selected_text" in df
        self.bpe = bpe
        self.vocab = vocab
        self.max_len = max_len

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        data["tweet"] = row.text
        data["sentiment"] = row.sentiment

        if self.bpe is None or self.vocab is None:
            data["ids"] = torch.zeros(self.max_len, dtype=torch.long)
            data["masks"] = torch.zeros(self.max_len, dtype=torch.long)
            data["tweets_encoded"] = ""
            if self.labeled:
                data["selected_tweet"] = row.selected_text
                data["start_idx"] = torch.tensor(0, dtype=torch.long)
                data["end_idx"] = torch.tensor(0, dtype=torch.long)
            return data

        ids, masks, tweets_encoded = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweets_encoded"] = tweets_encoded

        if self.labeled:
            data["selected_tweet"] = row.selected_text
            start_idx, end_idx = self.get_target_idx(row, tweets_encoded)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx
        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        normalized_tweets = normalizeTweet(row.text)
        normalized_tweets = " " + " ".join(normalized_tweets.split())
        tweets_encoded = self.bpe.encode(normalized_tweets)

        encoding_ids = (
            self.vocab.encode_line(
                tweets_encoded, append_eos=False, add_if_not_exist=False
            )
            .long()
            .tolist()
        )

        sentiment_id = (
            self.vocab.encode_line(
                self.bpe.encode(row.sentiment), append_eos=False, add_if_not_exist=False
            )
            .long()
            .tolist()
        )

        ids = [0] + sentiment_id + [2, 2] + encoding_ids + [2]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
        else:
            ids = ids[: self.max_len]

        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != 1).long()
        return ids, masks, tweets_encoded

    def get_target_idx(self, row, tweets_encoded):
        normalized_selected_tweets = normalizeTweet(row.selected_text)
        normalized_selected_tweets = " " + " ".join(normalized_selected_tweets.split())
        normalized_tweets = normalizeTweet(row.text)
        normalized_tweets = " " + " ".join(normalized_tweets.split())

        len_st = len(normalized_selected_tweets) - 1
        idx0 = None
        idx1 = None

        if len(normalized_selected_tweets) > 1:
            for ind in (
                i
                for i, e in enumerate(normalized_tweets)
                if e == normalized_selected_tweets[1]
            ):
                if (
                    " " + normalized_tweets[ind : ind + len_st]
                    == normalized_selected_tweets
                ):
                    idx0 = ind
                    idx1 = ind + len_st - 1
                    break

        if idx0 is None and len(normalized_selected_tweets.split()) > 1:
            normalized_selected_tweets_1 = " " + " ".join(
                normalized_selected_tweets.split()[1:]
            )
            len_st_1 = len(normalized_selected_tweets_1) - 1
            for ind in (
                i
                for i, e in enumerate(normalized_tweets)
                if e == normalized_selected_tweets_1[1]
            ):
                if (
                    " " + normalized_tweets[ind : ind + len_st_1]
                    == normalized_selected_tweets_1
                ):
                    idx0 = ind
                    idx1 = ind + len_st_1 - 1
                    break

        if idx0 is None and len(normalized_selected_tweets.split()) > 1:
            normalized_selected_tweets_2 = " " + " ".join(
                normalized_selected_tweets.split()[:-1]
            )
            len_st_2 = len(normalized_selected_tweets_2) - 1
            for ind in (
                i
                for i, e in enumerate(normalized_tweets)
                if e == normalized_selected_tweets_2[1]
            ):
                if (
                    " " + normalized_tweets[ind : ind + len_st_2]
                    == normalized_selected_tweets_2
                ):
                    idx0 = ind
                    idx1 = ind + len_st_2 - 1
                    break

        if idx0 is None and len(normalized_selected_tweets.split()) > 2:
            normalized_selected_tweets_3 = " " + " ".join(
                normalized_selected_tweets_2.split()[:-1]
            )
            len_st_3 = len(normalized_selected_tweets_3) - 1
            for ind in (
                i
                for i, e in enumerate(normalized_tweets)
                if e == normalized_selected_tweets_3[1]
            ):
                if (
                    " " + normalized_tweets[ind : ind + len_st_3]
                    == normalized_selected_tweets_3
                ):
                    idx0 = ind
                    idx1 = ind + len_st_3 - 1
                    break

        sum_tot = -1
        flag = 0
        if idx0 is not None and idx1 is not None:
            for i, token in enumerate(tweets_encoded.split()):
                if "@@" not in token:
                    sum_tot += len(token) + 1
                else:
                    sum_tot += len(token) - 2
                if sum_tot >= idx0 and flag == 0:
                    start_idx = i
                    flag = 1
                if sum_tot >= idx1:
                    end_idx = i
                    break
        else:
            start_idx = 0
            end_idx = 0

        return start_idx + 4, end_idx + 4




## === cell 6
DL_NUM_WORKERS = 0


def get_train_val_loaders(df, train_idx, val_idx, batch_size=32):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df, bpe, vocab),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, bpe, vocab),
        batch_size=batch_size,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}
    return dataloaders_dict




## === cell 7
class BERTweetModel(nn.Module):
    def __init__(self, conf):
        super(BERTweetModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            MODEL_DIR, config=conf, local_files_only=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(conf.hidden_size * 4, 2)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        h = out.hidden_states
        x = torch.cat([h[-1], h[-2], h[-3], h[-4]], dim=-1)
        x = self.fc(self.dropout(x))
        start_logits, end_logits = x.split(1, -1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 8
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss




## === cell 9
def get_selected_text(tweets_encoded, start_idx, end_idx):
    selected_text = ""
    for token in tweets_encoded.split()[start_idx - 4 : end_idx - 3]:
        selected_text += " " + token
    selected_text = re.sub("@@ ", "", selected_text)
    selected_text = re.sub("@@", "", selected_text)
    selected_text = selected_text.replace("Ġ", " ")
    return " ".join(selected_text.split())


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def compute_jaccard_score(tweets_encoded, start_idx, end_idx, start_logits, end_logits):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    length = len(tweets_encoded.split())
    if start_pred < 4:
        start_pred = 4
    if end_pred > 3 + length:
        end_pred = 3 + length
    if start_pred > end_pred:
        start_pred = 4
        end_pred = 3 + length
        pred = get_selected_text(tweets_encoded, start_pred, end_pred).strip()
    else:
        pred = get_selected_text(tweets_encoded, start_pred, end_pred).strip()
    true = get_selected_text(tweets_encoded, start_idx, end_idx).strip()
    return jaccard(true, pred)




## === cell 10
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_check = 1000
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0
            for count, data in enumerate(dataloaders_dict[phase]):
                if count % 100 == 0:
                    print(count)

                ids = data["ids"].to(device)
                masks = data["masks"].to(device)
                tweets_encoded = data["tweets_encoded"]

                start_idx = data["start_idx"].to(device)
                end_idx = data["end_idx"].to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    start_logits, end_logits = model(ids, masks)
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * ids.size(0)

                start_idx_np = start_idx.detach().cpu().numpy()
                end_idx_np = end_idx.detach().cpu().numpy()
                start_logits_np = (
                    torch.softmax(start_logits, dim=1).detach().cpu().numpy()
                )
                end_logits_np = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

                for i in range(ids.size(0)):
                    epoch_jaccard += compute_jaccard_score(
                        tweets_encoded[i],
                        start_idx_np[i],
                        end_idx_np[i],
                        start_logits_np[i],
                        end_logits_np[i],
                    )

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

            print(
                "Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}".format(
                    epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard
                )
            )

        if epoch_loss < loss_check:
            loss_check = epoch_loss
            print("Saving model")
            torch.save(model.state_dict(), filename)
        elif epoch > 1:
            print("Training stopping")
            break




## === cell 11
num_epochs = 10
batch_size = 32
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)




## === cell 12
def run(fold):
    if not USE_TRANSFORMERS:
        raise RuntimeError(
            "Training requires local RoBERTa assets; running in baseline mode."
        )

    train_df = (
        pd.read_csv(os.path.join(DATA_DIR, "train.csv")).dropna().reset_index(drop=True)
    )
    train_df["text"] = train_df["text"].astype(str)
    train_df["selected_text"] = train_df["selected_text"].astype(str)

    (train_idx, val_idx) = list(skf.split(train_df, train_df.sentiment))[fold]
    print(f"Fold: {fold}")
    model = BERTweetModel(conf=config)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    criterion = loss_fn
    dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)
    print("starting training")
    train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        num_epochs,
        os.path.join(WORK_DIR, f"roberta_fold{fold}.pth"),
    )




## === cell 13
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df, bpe, vocab),
        batch_size=batch_size,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return loader




## === cell 14
def postprocessing(pred, tweet):
    pred_wo_spaces = "".join(pred.split())
    if len(pred_wo_spaces) == 0:
        return tweet
    length = len(pred_wo_spaces)
    flag = 0
    if len(tweet) > 0 and tweet[-1] == "@":
        return tweet
    else:
        for index, value in enumerate(tweet):
            count = 0
            letter = pred_wo_spaces[count]
            if value == letter:
                start_idx = index
                end_idx = index
                count += 1
                end_idx += 1
                while end_idx < len(tweet) and count < length:
                    if tweet[end_idx] == " ":
                        end_idx += 1
                    elif (
                        tweet[end_idx] in ["!", ".", "*", "-", "?"]
                        and pred_wo_spaces[count] != tweet[end_idx]
                    ):
                        end_idx += 1
                    elif tweet[end_idx] == pred_wo_spaces[count]:
                        end_idx += 1
                        count += 1
                    else:
                        break
                    if count == length:
                        flag = 1
                        break
                if flag == 1:
                    break
        if flag == 1:
            while start_idx > 0:
                if tweet[start_idx - 1] == " ":
                    break
                else:
                    start_idx = start_idx - 1
            while end_idx < len(tweet) - 1:
                if tweet[end_idx] == " ":
                    break
                else:
                    end_idx = end_idx + 1
            return tweet[start_idx:end_idx]
        elif "HTTPURL" in pred.split():
            return tweet
        elif "@USER" in pred.split():
            return tweet
        else:
            return tweet




## === cell 15
_POS_WORDS = {
    "good",
    "great",
    "love",
    "best",
    "amazing",
    "awesome",
    "happy",
    "nice",
    "wonderful",
    "excellent",
    "fantastic",
    "cool",
    "enjoy",
    "fun",
    "beautiful",
    "thank",
    "thanks",
    "glad",
    "yay",
    "congrats",
    "sweet",
    "brilliant",
    "perfect",
    "excited",
    "positive",
    "smile",
    "lol",
    "haha",
    "lmao",
    "like",
    "liked",
    "lovely",
    "adorable",
}
_NEG_WORDS = {
    "bad",
    "worst",
    "hate",
    "awful",
    "sad",
    "angry",
    "terrible",
    "horrible",
    "sucks",
    "suck",
    "sorry",
    "pain",
    "annoying",
    "disappointed",
    "disappointing",
    "ugh",
    "mad",
    "upset",
    "negative",
    "cry",
    "stupid",
    "fail",
    "wtf",
    "broke",
    "broken",
    "tired",
    "sick",
    "unhappy",
    "ruined",
    "trash",
    "lame",
}


def _clean_token_for_match(t: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", "", str(t).lower())


def _token_emphasis_score(t: str) -> int:
    t = "" if t is None else str(t)
    s = 0
    if "!" in t:
        s += 2
    if "?" in t:
        s += 1
    if re.search(r"(.)\1\1+", t.lower()):  # elongated chars
        s += 2
    if t.isupper() and len(_clean_token_for_match(t)) >= 3:
        s += 1
    return s


def heuristic_selected_text(tweet: str, sentiment: str) -> str:
    tweet = "" if tweet is None else str(tweet)
    sentiment = "" if sentiment is None else str(sentiment).lower().strip()

    tw = tweet.strip()
    if not tw:
        return ""

    if sentiment == "neutral":
        return tw

    toks = tw.split()
    if not toks:
        return ""

    target_lex = _POS_WORDS if sentiment == "positive" else _NEG_WORDS

    best = None  # (score, i, j)
    n = len(toks)
    max_window = 8

    for i in range(n):
        lex_hits = 0
        emph_sum = 0
        for j in range(i, min(n, i + max_window)):
            w = toks[j]
            cw = _clean_token_for_match(w)
            if cw in target_lex:
                lex_hits += 1
            emph_sum += _token_emphasis_score(w)

            span_len = j - i + 1
            score = lex_hits * 10 + emph_sum * 2 - span_len

            if lex_hits == 0 and emph_sum == 0:
                continue

            if best is None or score > best[0]:
                best = (score, i, j)

    if best is not None:
        i, j = best[1], best[2]
        return " ".join(toks[i : j + 1]).strip()

    if sentiment == "negative":
        return toks[-1].strip()
    return toks[0].strip()




## === cell 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_df["text"] = test_df["text"].astype(str)

predictions = []

if USE_TRANSFORMERS:
    test_loader = get_test_loader(test_df, batch_size=batch_size)

    models = []
    for fold in range(skf.n_splits):
        ckpt_candidates = [
            f"/kaggle/input/mosh1-data-orig/roberta_fold{fold}.pth",
            os.path.join(WORK_DIR, f"roberta_fold{fold}.pth"),
        ]
        ckpt_path = None
        for c in ckpt_candidates:
            if os.path.isfile(c):
                ckpt_path = c
                break

        if ckpt_path is not None:
            m = BERTweetModel(conf=config).to(device)
            state = torch.load(ckpt_path, map_location="cpu")
            m.load_state_dict(state, strict=True)
            m.eval()
            models.append(m)

    if len(models) == 0:
        run(0)
        ckpt_path = os.path.join(WORK_DIR, "roberta_fold0.pth")
        m = BERTweetModel(conf=config).to(device)
        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)

    for data in test_loader:
        ids = data["ids"].to(device)
        masks = data["masks"].to(device)
        tweets_encoded = data["tweets_encoded"]
        tweet = data["tweet"]

        start_logits_list = []
        end_logits_list = []
        for model in models:
            with torch.no_grad():
                output = model(ids, masks)
                start_logits_list.append(
                    torch.softmax(output[0], dim=1).detach().cpu().numpy()
                )
                end_logits_list.append(
                    torch.softmax(output[1], dim=1).detach().cpu().numpy()
                )

        start_logits = np.mean(start_logits_list, axis=0)
        end_logits = np.mean(end_logits_list, axis=0)

        for i in range(ids.size(0)):
            start_pred = int(np.argmax(start_logits[i]))
            end_pred = int(np.argmax(end_logits[i]))
            length = len(tweets_encoded[i].split())
            if start_pred < 4:
                start_pred = 4
            if end_pred > 3 + length:
                end_pred = 3 + length
            if start_pred > end_pred:
                start_pred = 4
                end_pred = 3 + length

            pred = get_selected_text(tweets_encoded[i], start_pred, end_pred).strip()
            try:
                pred = postprocessing(pred, tweet[i].strip())
            except Exception:
                pred = tweet[i]
            predictions.append(pred)
else:
    predictions = [
        heuristic_selected_text(t, s)
        for t, s in zip(
            test_df["text"].fillna("").astype(str).tolist(),
            test_df["sentiment"].fillna("").astype(str).tolist(),
        )
    ]

if len(predictions) != len(test_df):
    raise RuntimeError(
        f"Predictions length mismatch: {len(predictions)} vs test {len(test_df)}"
    )

print("Predictions:", len(predictions), "rows")



## === cell 17
sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

pred_df = pd.DataFrame(
    {"textID": test_df["textID"].values, "selected_text": predictions}
)
sub_df = sub_df[["textID"]].merge(pred_df, on="textID", how="left")

sub_df["selected_text"] = sub_df["selected_text"].fillna("")

out_path = os.path.join(WORK_DIR, "submission.csv")
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub_df.head())
