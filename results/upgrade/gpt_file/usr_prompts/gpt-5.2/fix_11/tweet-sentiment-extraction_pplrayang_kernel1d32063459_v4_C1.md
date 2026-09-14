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

0.7140237092971802

# 6. Current score

0.59516

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fix the protobuf/transformers initialization crash by avoiding the forced pure-Python protobuf backend (it is incompatible with the current protobuf runtime in this environment). Then I make checkpoint discovery robust: if the external `/kaggle/input/roberta714kernel/` dataset isn’t present, the code fall back to a safe, valid baseline prediction that still produces a correctly formatted `submission.csv`. This keeps the core model/inference logic intact when checkpoints exist, but guarantees an end-to-end run and a valid submission file when they don’t. Finally, I ensure predictions always align 1:1 with the sample submission rows.'
- What this solution (achieved 0.59324) has done: 'I fix the crash caused by an incompatibility between `transformers` and the installed `protobuf` by disabling protobuf-dependent parts of `transformers` (so the import works in this environment). Then I keep your existing “use external checkpoints if present, otherwise fallback” logic intact, but make the tokenizer/model imports lazy so the baseline path never touches `transformers/tokenizers` when checkpoints are missing. Finally, I ensure the script always writes a valid `submission.csv` with the required columns and 1:1 row alignment.'
- What this solution (achieved 0.585) has done: 'Your current score (0.59324) is below the target (0.7140), and the main issue is that when the external RoBERTa checkpoints aren’t available your fallback predicts the full tweet for every sentiment, which is a weak baseline for this competition. I keep your overall pipeline intact (same I/O, same inference path when checkpoints exist), but improve only the fallback path with a simple rule-based extractor that is known to score substantially higher on this dataset: return full text for neutral, and for positive/negative return the “strongest” phrase using a small sentiment lexicon and negation handling. This should move the score upward toward the target while staying within Kaggle constraints and still producing a valid `submission.csv`. No training loops, model architecture, or metric semantics are changed.'
- What this solution (achieved 0.5856) has done: 'Your current score (0.585) is well below the target (0.7140), so we should improve accuracy while keeping the model/inference logic unchanged. The biggest safe gain with minimal change is to (1) use the official competition-style “whitespace-cleaned” text for baseline extraction (your current fallback uses the raw text, which mismatches the evaluation semantics), and (2) strengthen the fallback extractor using a tiny, deterministic phrase-span scorer (still rule-based, no training, no architecture changes). I only touch the `not HAS_EXTERNAL_CKPT` path and keep the external checkpoint path identical. This should move the fallback submission noticeably upward toward the target without altering core model code.'
- What this solution (achieved 0.59518) has done: 'Your current score (0.5856) is below the target (0.7140), so we should cautiously improve accuracy while keeping the model path unchanged. The biggest low-risk gain here is in the fallback (no external checkpoints) extractor: it currently guesses spans from a tiny lexicon without learning from the provided training labels. I keep the same overall pipeline and only upgrade the fallback to a deterministic, label-driven phrase scorer built from `train.csv` (PMI-style token association per sentiment) and then select the best contiguous span by token scores; this usually yields a sizable Jaccard lift without changing any neural architecture/training loops. I also ensure the fallback returns an exact substring of the normalized tweet (matching the competition’s whitespace semantics) to avoid Jaccard penalties from formatting mismatches.'
- What this solution (achieved 0.59516) has done: 'Your current score (0.59518) is well below the target (0.7140), so we should improve accuracy while keeping your existing model path untouched. Since external checkpoints are typically missing, the only safe lever is the fallback span selector: I keep your PMI-style token scoring, but switch the span search to a deterministic maximum-subarray (Kadane) over token scores with small edge penalties to avoid returning full tweet too often. I also add two tiny competition-specific rules that are known to help Jaccard without changing semantics: for neutral always return the full cleaned text, and for very short tweets (≤2 tokens) return the full text to avoid over-fragmentation. These are minimal changes confined to the `not HAS_EXTERNAL_CKPT` path and should move the score upward toward the target band.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import sys
import re
import glob
import random
import warnings
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm

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

INPUT_BASE = "/kaggle/input/tweet-sentiment-extraction/"
test_file = os.path.join(INPUT_BASE, "test.csv")
submission_template = os.path.join(INPUT_BASE, "sample_submission.csv")
train_file = os.path.join(INPUT_BASE, "train.csv")

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

CLS_TOK = 0
PAD_TOK = 1
SEP_TOK = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if len(state_dict) == 0:
        return state_dict
    k0 = next(iter(state_dict.keys()))
    if isinstance(k0, str) and k0.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


outdir = "/kaggle/input/roberta714kernel/"

_CANDIDATE_TOKENIZER_DIRS = [
    outdir,
    os.path.join(outdir, "roberta-base"),
    os.path.join(outdir, "model"),
    os.path.join(outdir, "models"),
]
_CANDIDATE_MODEL_DIRS = _CANDIDATE_TOKENIZER_DIRS.copy()


def _find_first_existing_file(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


MODEL_VOCAB_PATH = _find_first_existing_file(
    [os.path.join(d, "vocab.json") for d in _CANDIDATE_TOKENIZER_DIRS]
)
MODEL_VOCAB_MERGES_PATH = _find_first_existing_file(
    [os.path.join(d, "merges.txt") for d in _CANDIDATE_TOKENIZER_DIRS]
)

MODEL_CONFIG_PATH = _find_first_existing_file(
    [os.path.join(d, "config.json") for d in _CANDIDATE_MODEL_DIRS]
)
ROBERTA_PATH = _find_first_existing_file(_CANDIDATE_MODEL_DIRS)

HAS_EXTERNAL_CKPT = (
    os.path.isdir(outdir)
    and MODEL_VOCAB_PATH is not None
    and MODEL_VOCAB_MERGES_PATH is not None
    and MODEL_CONFIG_PATH is not None
    and ROBERTA_PATH is not None
)

print("DEVICE:", DEVICE)
print("Checkpoint dataset present:", HAS_EXTERNAL_CKPT)
if HAS_EXTERNAL_CKPT:
    print("Using ROBERTA_PATH:", ROBERTA_PATH)
    print("Using MODEL_CONFIG_PATH:", MODEL_CONFIG_PATH)
    print("Using vocab:", MODEL_VOCAB_PATH)
    print("Using merges:", MODEL_VOCAB_MERGES_PATH)
else:
    print(
        f"WARNING: External checkpoints not found at {outdir}. "
        "Will generate a valid baseline submission without model inference."
    )



## === cell 1
_tokenizers = None


def _get_tokenizers():
    global _tokenizers
    if _tokenizers is None:
        import tokenizers as _tk

        _tokenizers = _tk
    return _tokenizers


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns

        tk = _get_tokenizers()
        self.tokenizer = tk.ByteLevelBPETokenizer(
            MODEL_VOCAB_PATH,
            MODEL_VOCAB_MERGES_PATH,
            lowercase=True,
            add_prefix_space=True,
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
        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(str(row.sentiment)).ids

        ids = [CLS_TOK] + sentiment_id + [SEP_TOK, SEP_TOK] + encoding.ids + [SEP_TOK]
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

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

        for ind in (
            i
            for i, e in enumerate(tweet)
            if len(selected_text) > 1 and e == selected_text[1]
        ):
            if " " + tweet[ind : ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        offsets_list = offsets.tolist() if torch.is_tensor(offsets) else offsets
        for j, (offset1, offset2) in enumerate(offsets_list):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        if len(target_idx) == 0:
            return 0, 0

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx




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
_RobertaModel = None
_RobertaConfig = None


def _get_roberta_classes():
    global _RobertaModel, _RobertaConfig
    if _RobertaModel is None or _RobertaConfig is None:
        from transformers import RobertaModel, RobertaConfig

        _RobertaModel = RobertaModel
        _RobertaConfig = RobertaConfig
    return _RobertaModel, _RobertaConfig


class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        RobertaModel, RobertaConfig = _get_roberta_classes()
        config = RobertaConfig.from_pretrained(
            MODEL_CONFIG_PATH, output_hidden_states=True
        )
        self.roberta = RobertaModel.from_pretrained(ROBERTA_PATH, config=config)
        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
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




## === cell 5
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)
test_df["sentiment"] = test_df["sentiment"].astype(str)

predictions = []

if not HAS_EXTERNAL_CKPT:

    def _clean_text_like_training(text: str) -> str:
        text = "" if pd.isna(text) else str(text)
        return " " + " ".join(text.lower().split())

    def _tokenize_with_spans(text):
        return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def _clean_token(w: str) -> str:
        w = w.lower()
        w = re.sub(r"^[^\w']+|[^\w']+$", "", w)
        return w

    train_df = pd.read_csv(train_file, usecols=["text", "selected_text", "sentiment"])
    train_df["text"] = train_df["text"].astype(str)
    train_df["selected_text"] = train_df["selected_text"].astype(str)
    train_df["sentiment"] = train_df["sentiment"].astype(str)

    tok_sent_counts = {
        "positive": Counter(),
        "negative": Counter(),
        "neutral": Counter(),
    }
    tok_all_counts = Counter()
    sent_totals = Counter()

    for t, st, s in zip(
        train_df["text"], train_df["selected_text"], train_df["sentiment"]
    ):
        s = str(s).lower().strip()
        if s not in tok_sent_counts:
            continue
        st_clean = _clean_text_like_training(st)
        toks = [_clean_token(x[0]) for x in _tokenize_with_spans(st_clean)]
        toks = [w for w in toks if w != ""]
        if not toks:
            continue
        sent_totals[s] += len(toks)
        tok_sent_counts[s].update(toks)
        tok_all_counts.update(toks)

    vocab = list(tok_all_counts.keys())
    V = max(len(vocab), 1)
    total_all = sum(tok_all_counts.values()) if len(tok_all_counts) else 1

    alpha = 1.0

    def _token_score(token: str, sentiment: str) -> float:
        if token == "":
            return 0.0
        c_ts = tok_sent_counts[sentiment][token]
        c_t = tok_all_counts[token]
        p_t_s = (
            (c_ts + alpha) / (sent_totals[sentiment] + alpha * V)
            if sent_totals[sentiment] > 0
            else (alpha / (alpha * V))
        )
        p_t = (c_t + alpha) / (total_all + alpha * V)
        return math.log(p_t_s / p_t)

    def _is_noise_token(raw: str) -> bool:
        r = raw.lower()
        return bool(re.match(r"^(https?://|www\.|@)", r))

    def baseline_selected_text(text, sentiment):
        text_clean = _clean_text_like_training(text)
        sent = "" if pd.isna(sentiment) else str(sentiment).lower().strip()
        if text_clean.strip() == "":
            return ""

        if sent == "neutral":
            return text_clean.strip()

        toks = _tokenize_with_spans(text_clean)
        if not toks:
            return text_clean.strip()

        if len(toks) <= 2:
            return text_clean.strip()

        scores = []
        for raw, a, b in toks:
            if _is_noise_token(raw):
                scores.append(-0.6)
                continue
            w = _clean_token(raw)
            if w == "":
                scores.append(-0.2)
                continue
            scores.append(_token_score(w, sent))

        best_sum = -1e18
        best_i = 0
        best_j = 0
        cur_sum = -1e18
        cur_i = 0

        len_pen = 0.04
        base_pen = 0.10

        for j in range(len(scores)):
            val = scores[j] - len_pen

            if cur_sum < 0:
                cur_sum = val
                cur_i = j
            else:
                cur_sum += val

            adj_sum = cur_sum - base_pen
            if adj_sum > best_sum:
                best_sum = adj_sum
                best_i = cur_i
                best_j = j

        if best_sum <= 0.0:
            k = int(np.argmax(scores))
            if scores[k] <= 0.20:
                return text_clean.strip()
            a, b = toks[k][1], toks[k][2]
            cand = text_clean[a:b].strip()
            return cand if cand else text_clean.strip()

        start = toks[best_i][1]
        end = toks[best_j][2]
        cand = text_clean[start:end].strip()
        return cand if cand else text_clean.strip()

    predictions = [
        baseline_selected_text(t, s)
        for t, s in zip(test_df["text"].tolist(), test_df["sentiment"].tolist())
    ]
else:
    test_loader = get_test_loader(test_df, batch_size=batch_size)

    models = []
    ckpt_files = sorted(glob.glob(os.path.join(outdir, "*.pth")))
    if len(ckpt_files) == 0:
        raise FileNotFoundError(
            f"No .pth checkpoints found under {outdir}. Available files: {sorted(os.listdir(outdir))[:50]}"
        )

    print(f"Found {len(ckpt_files)} checkpoint(s). Loading models..")
    for ckpt_path in ckpt_files:
        model = TweetModel().to(DEVICE)

        state = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_module_prefix(state)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) or len(unexpected):
            print(
                f"Checkpoint {os.path.basename(ckpt_path)} loaded with missing={len(missing)} unexpected={len(unexpected)}"
            )

        model.eval()
        print(f"Loaded {ckpt_path}")
        models.append(model)

    if len(models) == 0:
        raise RuntimeError("No models loaded; cannot run inference.")

    for data in tqdm(test_loader, total=len(test_loader)):
        ids = data["ids"].to(DEVICE)
        masks = data["masks"].to(DEVICE)
        tweet = data["tweet"]
        offsets = data["offsets"].cpu().numpy()

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

        for i in range(len(ids)):
            start_pred = int(np.argmax(start_logits[i]))
            end_pred = int(np.argmax(end_logits[i]))
            if start_pred > end_pred:
                pred = tweet[i]
            else:
                pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
            predictions.append(pred)

print("Generated predictions:", len(predictions))



## === cell 6
sub_df = pd.read_csv(submission_template)

if len(predictions) != len(sub_df):
    raise ValueError(
        f"Predictions length {len(predictions)} != submission length {len(sub_df)}"
    )

sub_df["selected_text"] = pd.Series(predictions, index=sub_df.index).astype(str)

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

print(sub_df.head(20))
print("Wrote submission.csv with shape:", sub_df.shape)
print("submission.csv saved at:", os.path.abspath("submission.csv"))
