# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7150368094444275

# 6. Current score

0.64263

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.48895) has done: 'I fix the two blockers preventing an end-to-end run: (1) the early `MessageFactory.GetPrototype` crash (a known protobuf/version issue triggered by `transformers` imports in some Kaggle images) by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing transformers, and (2) the missing RoBERTa vocab/merges/config/model files by adding a robust fallback that uses the standard `roberta-base` tokenizer/model via `from_pretrained` when the custom `/kaggle/input/robertamodel0524/` directory is absent. These changes preserve the same architecture (RoBERTa + hidden-state averaging + linear head) and the same inference logic, but ensure the code can actually load a model/tokenizer and generate predictions. Finally, I ensure `predictions` is always defined (so cell 6 doesn’t crash) and that `submission.csv` is written with the required columns and row count.'
- What this solution (achieved 0.48895) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation **and** unsetting the C++ implementation env var before any `transformers` import, plus ensuring the runtime uses a compatible protobuf module without changing your model/inference logic. I also correct the custom RoBERTa weight loading: `RobertaModel.from_pretrained` must receive a directory or model name, not a `.bin` file path; this was silently breaking the intended model loading and hurting score. Finally, I keep your existing architecture and prediction logic unchanged, but make the model/tokenizer loading paths robust so the notebook runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.48895) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before importing `transformers`, which is the actual trigger for this error in many Kaggle images. I also make the fold-weight detection robust: if the provided `/kaggle/input/robertalineardropout/` directory is missing (common in your file listing), the code cleanly fall back to a single `roberta-base` model without crashing, while keeping the same model architecture and inference logic. Finally, I keep your submission writing unchanged but ensure the pipeline always reaches it and produces a valid `submission.csv`. These changes should restore correct execution and, when fold weights are available, improve score substantially toward the target (your current 0.48895 is far below 0.715).'
- What this solution (achieved 0.44901) has done: 'I fix the immediate crash coming from the protobuf/transformers incompatibility by forcing the pure-Python protobuf implementation *and* setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing `transformers` (this is the most reliable way to avoid the `MessageFactory.GetPrototype` error in Kaggle images). I also make the fallback path score-healthy by (when fold weights are missing) loading a QA-style RoBERTa head (`RobertaForQuestionAnswering`) instead of an untrained randomly-initialized linear head; this preserves the same “RoBERTa → start/end logits → argmax span” inference semantics but avoids catastrophic performance from random weights. Finally, I keep your submission formatting the same, ensuring the script always runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.59324) has done: 'I fix the protobuf/transformers crash by forcing the pure-Python protobuf implementation *before any transformers import* and by importing `google.protobuf` early to lock the implementation (this removes the `MessageFactory.GetPrototype` runtime error). I also add a safe fallback so the script can still run even if transformers cannot be imported for some reason, producing a valid `submission.csv` (score-neutral vs a crash). To improve score toward your target without changing the core modeling approach, I switch the fallback model from an unrelated pretrained QA head to using your original span head only when fold weights exist, and otherwise default to selecting the whole tweet for neutral and for all cases where the model would be untrained (this avoids random span behavior that hurts Jaccard). Finally, I ensure the submission length and columns always match the template and the file is written with a `.csv` suffix.'
- What this solution (achieved 0.64263) has done: 'I fix the immediate protobuf/transformers crash (`MessageFactory.GetPrototype`) by forcing a compatible protobuf runtime before any `transformers` import and by pinning protobuf to the pure-Python implementation reliably. Then I fix the fold-weight path handling so the model branch actually runs when weights exist, and I keep your exact RoBERTa-span architecture/inference unchanged. Finally, I improve the “no-weights” fallback to a sentiment-aware heuristic (positive/negative pick best matching n-gram; neutral returns full text), which is a small, legitimate score improvement over always returning the whole tweet while still producing a valid `submission.csv`.'
- What this solution (achieved 0.64263) has done: 'I fix the protobuf/transformers crash by applying the environment variables *before any protobuf-related import* and by avoiding importing `google.protobuf` directly (it can lock in the broken implementation and still trigger `MessageFactory.GetPrototype`). Then I make the transformers import a bit more robust by catching any remaining exception and cleanly falling back to your existing sentiment-aware heuristic so the notebook always completes and writes `submission.csv`. This is a runtime/stability-only change that preserves your core model/inference logic when fold weights are available, and otherwise keeps your current heuristic behavior (so score changes should be minimal and generally non-negative due to actually running). Finally, I keep submission formatting identical while ensuring row alignment with `sample_submission.csv`.'
- What this solution (achieved 0.64263) has done: 'I fix the `MessageFactory.GetPrototype` crash by ensuring protobuf uses the pure-Python implementation *before* any indirect protobuf-triggering imports and by importing `google.protobuf` after setting env vars to lock the correct runtime. This is a runtime/stability fix that should let the transformers path work when available and prevent the notebook from failing in cell 1+. I also make the transformers import robust by falling back to the existing heuristic if the import still fails, preserving your current behavior and score when the model path is unusable. No core modeling logic, inference semantics, or submission formatting be changed.'
- What this solution (achieved 0.64263) has done: 'I fix the protobuf crash that prevents `transformers` from importing by proactively patching the missing `MessageFactory.GetPrototype` symbol (a known incompatibility in some Kaggle images) before importing `transformers`. This should allow your existing RoBERTa fold-weight inference path to run when weights are present, which is the main route to improving from your current heuristic-based score toward the 0.715 target. I also make the “custom roberta files” detection and tokenization offsets consistent (avoid hardcoded 4-offset padding when the sentiment token length varies), which is a correctness bug that can hurt span extraction even when the model runs. All changes are minimal, keep the same model/head/inference semantics, and still guarantee a valid `submission.csv` is written.'
- What this solution (achieved 0.64263) has done: 'Your current score (0.64263) is below the target (0.7150), so we should cautiously improve without changing the model/head or training approach. The biggest low-risk gain here is in inference post-processing: your span selection currently uses independent argmax for start/end, which often yields inconsistent or overly long spans and hurts word-level Jaccard. I keep the same logits and ensemble exactly as-is, but replace the argmax pair with the standard constrained best-span search (maximize start_logit + end_logit with end≥start and a max span length), plus a tiny neutral-sentiment rule (return full tweet) which is consistent with the task and commonly improves Jaccard. These changes only affect decoding/post-processing and should move your score upward toward the target band while keeping runtime within limits.'
- What this solution (achieved 0.64263) has done: 'Your current score (0.64263) is below the target (0.7150), so we should improve decoding/post-processing without changing the model, training, or feature extraction. The main low-risk gain is fixing a subtle but important bug: you currently run `softmax` over `dim=1` on tensors shaped `[batch, seq_len]`, which normalizes across the batch dimension instead of across tokens and harms span selection. I change that to `dim=1` → `dim=-1` (or `dim=1` after confirming shape) so probabilities are computed over the sequence length per example, keeping the exact same model outputs and ensemble. I also make the neutral rule slightly safer by returning the original raw text (not the lowercased/normalized version with a leading space) to better match punctuation/quotes expected by the metric, while keeping the overall semantics identical.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_CPP", None)

import sys
import re
import string
import random
import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn

import tokenizers

warnings.filterwarnings("ignore")

try:
    from google.protobuf import message_factory as _message_factory  # noqa: F401

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass


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
NUM_WORKERS = 2

ROBERTA_PATH = "/kaggle/input/robertamodel0524/"
MODEL_CONFIG_PATH = ROBERTA_PATH + "roberta-base-config.json"
MODEL_PATH = ROBERTA_PATH + "roberta-base-pytorch_model.bin"
MODEL_VOCAB_PATH = ROBERTA_PATH + "roberta-base-vocab.json"
MODEL_VOCAB_MERGES_PATH = ROBERTA_PATH + "roberta-base-merges.txt"
outdir = "/kaggle/input/robertalineardropout/"

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

CLS_TOK = 0
PAD_TOK = 1
SEP_TOK = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUSTOM_ROBERTA_FILES = os.path.isdir(ROBERTA_PATH) and all(
    os.path.exists(p)
    for p in [MODEL_CONFIG_PATH, MODEL_PATH, MODEL_VOCAB_PATH, MODEL_VOCAB_MERGES_PATH]
)

TRANSFORMERS_AVAILABLE = True
try:
    from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    _TRANSFORMERS_IMPORT_ERROR = repr(e)

from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)


def _join_path(d, f):
    return os.path.join(d, f)


FOLD_WEIGHT_PATHS = [
    _join_path(outdir, f"roberta_fold{fold+1}.pth") for fold in range(skf.n_splits)
]
HAS_FOLD_WEIGHTS = os.path.isdir(outdir) and all(
    os.path.exists(p) for p in FOLD_WEIGHT_PATHS
)

print("TRANSFORMERS_AVAILABLE:", TRANSFORMERS_AVAILABLE)
if not TRANSFORMERS_AVAILABLE:
    print(
        "Transformers import error (will fallback to heuristic submission):",
        _TRANSFORMERS_IMPORT_ERROR,
    )
print("USE_CUSTOM_ROBERTA_FILES:", USE_CUSTOM_ROBERTA_FILES)
print("HAS_FOLD_WEIGHTS:", HAS_FOLD_WEIGHTS)
print("device:", device)




## === cell 1
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns

        if not TRANSFORMERS_AVAILABLE:
            self.use_hf_fast = False
            self.hf_tokenizer = None
            self.tokenizer = None
            return

        self.use_hf_fast = not USE_CUSTOM_ROBERTA_FILES
        if self.use_hf_fast:
            self.hf_tokenizer = RobertaTokenizerFast.from_pretrained(
                "roberta-base", add_prefix_space=True
            )
        else:
            self.tokenizer = tokenizers.ByteLevelBPETokenizer(
                vocab=MODEL_VOCAB_PATH,
                merges=MODEL_VOCAB_MERGES_PATH,
                add_prefix_space=True,
                lowercase=True,
            )

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

        if self.use_hf_fast:
            sentiment_ids = self.hf_tokenizer.encode(
                str(row.sentiment),
                add_special_tokens=False,
            )
            enc = self.hf_tokenizer.encode_plus(
                tweet,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            tweet_ids = enc["input_ids"]
            tweet_offsets = enc["offset_mapping"]

            prefix_len = 1 + len(sentiment_ids) + 2  # [CLS] + sentiment + [SEP][SEP]
            ids = [CLS_TOK] + sentiment_ids + [SEP_TOK, SEP_TOK] + tweet_ids + [SEP_TOK]
            offsets = [(0, 0)] * prefix_len + list(tweet_offsets) + [(0, 0)]
        else:
            encoding = self.tokenizer.encode(tweet)
            sentiment_id = self.tokenizer.encode(str(row.sentiment)).ids

            prefix_len = 1 + len(sentiment_id) + 2
            ids = (
                [CLS_TOK] + sentiment_id + [SEP_TOK, SEP_TOK] + encoding.ids + [SEP_TOK]
            )
            offsets = [(0, 0)] * prefix_len + encoding.offsets + [(0, 0)]

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
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        if len(target_idx) == 0:
            return 0, 0

        return target_idx[0], target_idx[-1]




## === cell 2
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return loader




## === cell 3
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        if USE_CUSTOM_ROBERTA_FILES:
            config = RobertaConfig.from_pretrained(
                MODEL_CONFIG_PATH, output_hidden_states=True
            )
            self.roberta = RobertaModel.from_pretrained(ROBERTA_PATH, config=config)
        else:
            config = RobertaConfig.from_pretrained(
                "roberta-base", output_hidden_states=True
            )
            self.roberta = RobertaModel.from_pretrained("roberta-base", config=config)

        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hs = outputs.hidden_states

        x = torch.stack([hs[-1], hs[-2], hs[-3]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = self.fc(x)

        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits




## === cell 4
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
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


def decode_best_span(start_probs, end_probs, offsets, max_answer_len=30):
    L = len(start_probs)
    valid = np.array([(o1 != 0 or o2 != 0) for (o1, o2) in offsets], dtype=bool)
    if not valid.any():
        return 0, 0

    s = start_probs.copy()
    e = end_probs.copy()
    s[~valid] = -1e9
    e[~valid] = -1e9

    best_score = -1e18
    best_i, best_j = 0, 0
    for i in range(L):
        if not valid[i]:
            continue
        j_max = min(L - 1, i + max_answer_len - 1)
        j = i + int(np.argmax(e[i : j_max + 1]))
        score = s[i] + e[j]
        if score > best_score:
            best_score = score
            best_i, best_j = i, j
    return best_i, best_j




## === cell 5
POS_WORDS = {
    "good",
    "great",
    "love",
    "loved",
    "loving",
    "best",
    "awesome",
    "amazing",
    "nice",
    "happy",
    "glad",
    "fantastic",
    "wonderful",
    "excellent",
    "perfect",
    "thanks",
    "thank",
    "beautiful",
    "cool",
    "wow",
    "yay",
    "fun",
    "enjoy",
    "enjoyed",
}
NEG_WORDS = {
    "bad",
    "hate",
    "hated",
    "worst",
    "awful",
    "terrible",
    "sad",
    "angry",
    "upset",
    "annoying",
    "sucks",
    "suck",
    "ugh",
    "disappointed",
    "disappointing",
    "pain",
    "sorry",
    "worse",
    "boring",
    "mad",
    "hurt",
    "crap",
}

_word_re = re.compile(r"[A-Za-z']+")


def _tokenize_words(text):
    return _word_re.findall(str(text).lower())


def _best_ngram_span(text, lexicon, max_ngram=5):
    toks = str(text).split()
    if not toks:
        return str(text)

    best_i, best_j = 0, len(toks) - 1
    best_score = 0
    best_len = len(toks)

    for i in range(len(toks)):
        for j in range(i, min(len(toks), i + max_ngram)):
            span = " ".join(toks[i : j + 1])
            w = set(_tokenize_words(span))
            score = len(w.intersection(lexicon))
            if score > best_score or (
                score == best_score and (j - i + 1) < best_len and score > 0
            ):
                best_score = score
                best_len = j - i + 1
                best_i, best_j = i, j

    if best_score == 0:
        return str(text)
    return " ".join(toks[best_i : best_j + 1])


test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)

test_df["text_raw"] = test_df["text"].astype(str)

predictions = []

if (not TRANSFORMERS_AVAILABLE) or (not HAS_FOLD_WEIGHTS):
    print(
        "Using sentiment-aware heuristic predictions (no usable fold weights and/or transformers unavailable)."
    )
    for _, row in test_df.iterrows():
        txt = str(row["text_raw"])
        sent = str(row["sentiment"]).lower()
        if sent == "neutral":
            predictions.append(txt)
        elif sent == "positive":
            predictions.append(_best_ngram_span(txt, POS_WORDS, max_ngram=6))
        elif sent == "negative":
            predictions.append(_best_ngram_span(txt, NEG_WORDS, max_ngram=6))
        else:
            predictions.append(txt)
    print("Generated predictions:", len(predictions))
else:
    test_loader = get_test_loader(test_df, batch_size=batch_size)

    models = []
    print("loading models..")
    for fold in tqdm(range(skf.n_splits)):
        model = TweetModel().to(device)
        state = torch.load(
            _join_path(outdir, f"roberta_fold{fold+1}.pth"), map_location=device
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)

    for data in tqdm(test_loader):
        ids = data["ids"].to(device)
        masks = data["masks"].to(device)
        tweet = data[
            "tweet"
        ]  # normalized/lowercased with leading space (used for offsets)
        offsets = data["offsets"].cpu().numpy()

        start_logits = []
        end_logits = []
        for model in models:
            with torch.no_grad():
                out_start, out_end = model(ids, masks)

                start_logits.append(torch.softmax(out_start, dim=-1).cpu().numpy())
                end_logits.append(torch.softmax(out_end, dim=-1).cpu().numpy())

        start_logits = np.mean(start_logits, axis=0)
        end_logits = np.mean(end_logits, axis=0)

        base = len(predictions)
        for i in range(len(ids)):
            sent = str(test_df.iloc[base + i]["sentiment"]).lower()
            if sent == "neutral":
                predictions.append(str(test_df.iloc[base + i]["text_raw"]))
                continue

            s_pred, e_pred = decode_best_span(
                start_logits[i], end_logits[i], offsets[i], max_answer_len=30
            )
            if s_pred > e_pred:
                pred = str(test_df.iloc[base + i]["text_raw"])
            else:
                pred = get_selected_text(tweet[i], s_pred, e_pred, offsets[i])
            predictions.append(pred)

    print("Generated predictions:", len(predictions))



## === cell 6
sub_df = pd.read_csv(submission_template)

if len(predictions) != len(sub_df):
    raise RuntimeError(
        f"Predictions length {len(predictions)} != submission rows {len(sub_df)}"
    )

sub_df["selected_text"] = predictions

sub_df["selected_text"] = sub_df["selected_text"].astype(str)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(x.split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(x.split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(x.split()) == 1 else x
)

sub_df = sub_df[["textID", "selected_text"]]
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head(20).to_string(index=False))
print("\nWrote submission.csv with shape:", sub_df.shape)
