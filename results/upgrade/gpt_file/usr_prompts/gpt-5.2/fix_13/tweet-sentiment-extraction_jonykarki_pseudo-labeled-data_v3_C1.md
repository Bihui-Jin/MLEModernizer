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
joblib==1.5.2
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

0.6745771765708923

# 6. Current score

0.57828

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58579) has done: 'Your notebook fails because it assumes an offline RoBERTa/BERTweet model directory (tokenizer/config/weights) exists under `/kaggle/input`, but your provided file listing shows only the competition dataset files, so `MODEL_NAME_OR_DIR`, `tokenizer`, and `model_config` never get defined and everything downstream breaks. To make the pipeline run end-to-end and still produce a valid submission, I add a robust fallback path: if no local model/weights are found, skip neural inference and use a deterministic rule-based selector (neutral → full text; pos/neg → try to find the exact `selected_text`-like span using simple heuristics). This keeps the original core model code intact when model files are available, but guarantees a valid `submission.csv` is written in all cases. The fallback is score-oriented (better than empty/full-text baseline) without changing your model semantics when the model assets exist.'
- What this solution (achieved 0.57828) has done: 'Your current score (0.58579) is below the target (0.67458), so we should improve predictions while keeping your overall approach intact. The biggest low-risk gain is fixing the post-processing for the neural path: when `a > b` or the span is bad, returning the *entire tokenized tweet with leading space* hurts Jaccard; instead we should fall back to the original raw `text` (no added leading space) and also constrain spans to the actual tweet token region (token_type_ids==0) to avoid selecting sentiment/padding tokens. Additionally, your heuristic fallback can be strengthened slightly (still rule-based) by choosing the best substring via word-level Jaccard against sentiment-specific cue-phrases and by ensuring exact substring extraction from the original tweet, which typically improves over punctuation-splitting. These are minimal, metric-aligned changes that preserve the core model and inference logic and should move the score upward toward the target.'
- What this solution (achieved 0.57828) has done: 'Your current score (0.57828) is well below the target (0.67458), so we should improve predictions with minimal risk while keeping the same model and span-extraction approach. The biggest likely gain is fixing tweet-token bounds: with RoBERTa pairs, `token_type_ids` are often all zeros, so your current “tweet region” can accidentally include the sentiment side and special tokens; we instead detect the tweet span by locating the first and second `</s>` (id=2) in `input_ids` and clamp predictions to that region. We also make the fallback for invalid/empty spans metric-aligned by returning the cleaned raw tweet (not the tokenized one with a leading space) and apply a tiny, safe improvement: for neutral sentiment always return the full raw tweet. These changes preserve your core logic (same models, same averaging, same argmax span selection) but reduce common span mistakes that heavily hurt Jaccard.'
- What this solution (achieved 0.57828) has done: 'Your gap to target is sizable (0.57828 → 0.67458), so we should safely increase Jaccard without changing the model/training logic. The biggest low-risk gain here is fixing a mismatch between how you train/locate targets (token-based match inside the “tweet” segment) and how you decode predictions (token decode can lose exact punctuation/spacing), so we instead map predicted token indices back to the exact character span in the original tweet using offsets and then slice the raw text. We also improve the tweet token bounds detection for RoBERTa pair inputs by explicitly using the first `</s>` boundary, and we keep your existing neutral/full-text fallback behavior intact. These are inference/post-processing-only changes, so the core architecture and ensembling remain identical while span extraction becomes much more metric-aligned.'
- What this solution (achieved 0.57828) has done: 'Your score is below the target (0.57828 vs 0.67458), so we should improve span quality with minimal, inference-only changes. The biggest likely issue is that the offset mapping you use for raw slicing does not correspond to the *same* `input_ids_row` used to pick `(a,b)` (because you re-tokenize inside the loop), causing frequent misaligned/incorrect spans. I fix this by having the dataset return `offset_mapping` and using it directly (perfectly aligned), and by clamping `(a,b)` using offset-based “tweet token” bounds (tokens with non-empty offsets) rather than relying on `</s>` positions. Everything else (model, weights loading, ensembling, argmax span selection, neutral full-text rule, heuristic fallback when no model) stays the same.'
- What this solution (achieved 0.57828) has done: 'Your score (0.57828) is below the target (0.67458), so we should improve span extraction quality with minimal, inference-only changes. The biggest gain without changing the model is to decode the predicted span more robustly: instead of taking independent argmax for start/end, we select the best (start,end) pair under a max-length constraint (standard for this competition) while still using the same model outputs. We also clamp candidates to true tweet tokens using `offset_mapping` (non-(0,0) offsets) and apply a small, metric-aligned neutral rule: if sentiment is neutral or the tweet is very short, return the full raw tweet. These changes keep your architecture, weights, and ensembling identical while reducing common span mistakes that heavily hurt word-level Jaccard.'
- What this solution (achieved 0.57828) has done: 'Your score is below the target (0.57828 vs 0.67458), so the safest way to move upward without changing your model/training is to improve span decoding and a couple of metric-aligned post-processing rules. I keep the same model, same 8-fold weight ensembling, and the same start/end logits; the only change in “core inference” is adding a very small neutral/short-text rule (already present) plus a standard “full-text if sentiment token wins” rule that fixes a common failure mode in this competition. I also slightly adjust the joint span search to be log-prob based (numerically stable, same semantics) and remove the “drop last token if punctuation/stopword” heuristic, which often deletes sentiment-bearing words and hurts Jaccard. These are minimal inference-only changes and should increase score toward your target.'
- What this solution (achieved 0.57828) has done: 'Your current score (0.57828) is below the target (0.67458), so the smallest safe lift is to improve inference-time span decoding without changing your model, weights, or training approach. I keep your offset-based raw slicing, but fix two common extraction failure modes in this competition: (1) incorrectly allowing special/sentiment-side tokens into the candidate span set, and (2) selecting a span that maps to empty offsets (yielding full-text fallbacks). Concretely, I derive the “tweet token” bounds from the first contiguous non-zero-offset region (the actual tweet segment), add a tiny logit mask so start/end cannot land on (0,0) offsets, and add the standard “if best span is very low-confidence, return full tweet” guard (minimal and metric-aligned). These are inference-only changes and should move the score upward toward your target without altering the core architecture or ensembling.'

# 9. Code solution

## === cell 0
import os
import re
import json
import string
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import transformers

from nltk.corpus import stopwords


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

print("Transformers:", transformers.__version__)
print("Torch:", torch.__version__)



## === cell 1
LEARNING_RATE = 6e-5
MAX_LEN = 126
TRAIN_BATCH_SIZE = 35
VALID_BATCH_SIZE = 32
EPOCHS = 3

INPUT_PATH = "/kaggle/input/"

CANDIDATE_MODEL_DIRS = [
    os.path.join(INPUT_PATH, "twitroberta"),
    os.path.join(INPUT_PATH, "tweet-sentiment-extraction", "twitroberta"),
    os.path.join(INPUT_PATH, "bertweet-model", "BERTweet_base_transformers"),
    os.path.join(
        INPUT_PATH,
        "tweet-sentiment-extraction",
        "bertweet-model",
        "BERTweet_base_transformers",
    ),
]

REQUIRED_FOR_TOKENIZER = ["vocab.json", "merges.txt"]
REQUIRED_FOR_CONFIG = ["config.json"]


def _looks_like_roberta_dir(p: str) -> bool:
    if not os.path.isdir(p):
        return False
    has_tok = all(os.path.isfile(os.path.join(p, f)) for f in REQUIRED_FOR_TOKENIZER)
    has_cfg = any(os.path.isfile(os.path.join(p, f)) for f in REQUIRED_FOR_CONFIG)
    return has_tok and has_cfg


def _resolve_local_model_dir() -> str:
    for p in CANDIDATE_MODEL_DIRS:
        if _looks_like_roberta_dir(p):
            return p

    for root, dirs, files in os.walk(INPUT_PATH):
        file_set = set(files)
        if {"vocab.json", "merges.txt", "config.json"}.issubset(file_set):
            return root

    return ""


MODEL_DIR = _resolve_local_model_dir()
HAS_LOCAL_MODEL = bool(MODEL_DIR)
MODEL_NAME_OR_DIR = MODEL_DIR if HAS_LOCAL_MODEL else None

print("Resolved MODEL_NAME_OR_DIR:", MODEL_NAME_OR_DIR)
print("HAS_LOCAL_MODEL:", HAS_LOCAL_MODEL)

tokenizer = None
model_config = None

if HAS_LOCAL_MODEL:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_DIR,
        use_fast=True,
        local_files_only=True,
    )
    model_config = transformers.AutoConfig.from_pretrained(
        MODEL_NAME_OR_DIR,
        local_files_only=True,
    )
    model_config.output_hidden_states = True

try:
    stop_words = set(stopwords.words("english"))
except Exception:
    import nltk

    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))




## === cell 2
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = [" " + " ".join(str(tweet).split()) for tweet in tweets]
        self.sentiments = [
            " " + " ".join(str(sentiment).split()) for sentiment in sentiments
        ]
        self.selected_texts = [
            " " + " ".join(str(selected_text).split())
            for selected_text in selected_texts
        ]
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.tweets)

    def _sentiment_text(self, s):
        s = s.strip()
        if s != "neutral":
            return f"{s} {s}"
        return s

    def __getitem__(self, item):
        tweet = self.tweets[item]
        sentiment = self.sentiments[item].strip()
        selected = self.selected_texts[item]

        enc = tokenizer(
            tweet,
            self._sentiment_text(sentiment),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        token_type_ids = enc.get("token_type_ids", [0] * self.max_len)
        offsets = enc.get("offset_mapping", [(0, 0)] * self.max_len)

        start_index, end_index = 0, 0

        tweet_positions = [
            i
            for i, tti in enumerate(token_type_ids)
            if tti == 0 and attention_mask[i] == 1
        ]
        if len(tweet_positions) > 0:
            tweet_start = tweet_positions[0]
            tweet_end = tweet_positions[-1] + 1
        else:
            tweet_start, tweet_end = 0, self.max_len

        sel_ids = tokenizer(
            selected,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        if len(sel_ids) > 0:
            hay = input_ids[tweet_start:tweet_end]
            for j in range(0, max(0, len(hay) - len(sel_ids) + 1)):
                if hay[j : j + len(sel_ids)] == sel_ids:
                    start_index = tweet_start + j
                    end_index = tweet_start + j + len(sel_ids)  # exclusive
                    break

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "targets_start": torch.tensor(start_index, dtype=torch.long),
            "targets_end": torch.tensor(end_index, dtype=torch.long),
            "orig_tweet": tweet,
            "orig_selected": selected,
            "sentiment": sentiment,
        }




## === cell 3
class TweetModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.roberta = transformers.RobertaModel.from_pretrained(
            MODEL_NAME_OR_DIR,
            config=conf,
            local_files_only=True,
        )
        self.drop_out = nn.Dropout(0.1)
        self.activation = nn.LeakyReLU()
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.roberta(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 4
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values
print(df_test.head())
print("Test shape:", df_test.shape)



## === cell 5
if HAS_LOCAL_MODEL:
    test_dataset = TweetDataset(
        tweets=df_test.text.values,
        sentiments=df_test.sentiment.values,
        selected_texts=df_test.selected_text.values,
    )

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=VALID_BATCH_SIZE,
        num_workers=0,
    )
else:
    data_loader = None



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def _resolve_weight_path(rel_path: str) -> str:
    p1 = os.path.join(INPUT_PATH, rel_path)
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(INPUT_PATH, "tweet-sentiment-extraction", rel_path)
    if os.path.isfile(p2):
        return p2
    return p1  # keep original for warning message


def _load_model(weight_path: str):
    m = TweetModel(conf=model_config).to(device)
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location=device)
        m.load_state_dict(state, strict=True)
    else:
        print(
            f"WARNING: Missing weight file: {weight_path} (using randomly initialized weights)"
        )
    m.eval()
    return m


if HAS_LOCAL_MODEL:
    model1 = _load_model(_resolve_weight_path("twitroberta/model_0.bin"))
    model2 = _load_model(_resolve_weight_path("twitroberta/model_1.bin"))
    model3 = _load_model(_resolve_weight_path("twitroberta/model_2.bin"))
    model4 = _load_model(_resolve_weight_path("twitroberta/model_3.bin"))
    model5 = _load_model(_resolve_weight_path("twitroberta/model_4.bin"))
    model6 = _load_model(_resolve_weight_path("twitroberta/model_5.bin"))
    model7 = _load_model(_resolve_weight_path("twitroberta/model_6.bin"))
    model8 = _load_model(_resolve_weight_path("twitroberta/model_7.bin"))
else:
    model1 = model2 = model3 = model4 = model5 = model6 = model7 = model8 = None




## === cell 7
def _clean_spaces(s: str) -> str:
    return " ".join(str(s).split()).strip()


def _jaccard(a: str, b: str) -> float:
    a_set = set(_clean_spaces(a).lower().split())
    b_set = set(_clean_spaces(b).lower().split())
    if len(a_set) == 0 and len(b_set) == 0:
        return 1.0
    if len(a_set) == 0 or len(b_set) == 0:
        return 0.0
    return len(a_set & b_set) / float(len(a_set | b_set))


def _best_span_heuristic(text: str, sentiment: str) -> str:
    text0 = str(text)
    text_clean = _clean_spaces(text0)
    if text_clean == "":
        return ""

    s = str(sentiment).strip().lower()
    if s == "neutral":
        return text_clean

    words = []
    for m in re.finditer(r"\S+", text0):
        w = m.group(0)
        w_clean = re.sub(r"^[^\w']+|[^\w']+$", "", w.lower())
        words.append((m.start(), m.end(), w, w_clean))

    if not words:
        return text_clean

    neg_markers = {
        "not",
        "no",
        "never",
        "can't",
        "dont",
        "don't",
        "won't",
        "wont",
        "bad",
        "hate",
        "worst",
        "awful",
        "terrible",
        "boring",
        "annoying",
        "disappointed",
    }
    pos_markers = {
        "love",
        "great",
        "good",
        "best",
        "amazing",
        "awesome",
        "nice",
        "happy",
        "fantastic",
        "excellent",
        "cool",
        "wonderful",
        "perfect",
        "fun",
    }
    markers = neg_markers if s == "negative" else pos_markers

    toks_clean = [wc for (_, _, _, wc) in words if wc]
    present = [t for t in toks_clean if t in markers]
    ref = (
        " ".join(present[:6])
        if len(present) > 0
        else ("bad" if s == "negative" else "good")
    )

    best = text_clean
    best_score = -1e9

    max_span = 12
    n = len(words)
    for i in range(n):
        for j in range(i, min(n, i + max_span)):
            span_text = _clean_spaces(text0[words[i][0] : words[j][1]])
            if span_text == "":
                continue
            span_toks = [w[3] for w in words[i : j + 1] if w[3]]
            if not span_toks:
                continue

            marker_hits = sum(t in markers for t in span_toks)
            content = sum((t not in stop_words) for t in span_toks)

            jac = _jaccard(span_text, ref)

            length_penalty = abs(len(span_toks) - 3) / 3.0
            score = (
                2.2 * marker_hits + 0.12 * content + 1.8 * jac - 0.35 * length_penalty
            )

            if score > best_score:
                best_score = score
                best = span_text

    out = _clean_spaces(best)
    return out if out != "" else text_clean


def _get_tweet_token_bounds_from_offsets(offsets_row, mask_row):
    """
    Why (score): the pair encoding is [<s> tweet </s></s> sentiment </s> ...].
    Using "all non-zero offsets" can include the sentiment side for some tokenizers.
    We instead take the FIRST contiguous non-zero-offset region as the tweet segment,
    which prevents selecting sentiment tokens and usually improves Jaccard.
    """
    valid_len = int(sum(mask_row))

    non_zero = []
    for i in range(valid_len):
        s, e = int(offsets_row[i][0]), int(offsets_row[i][1])
        if not (s == 0 and e == 0):
            non_zero.append(i)

    if not non_zero:
        return 0, max(0, valid_len - 1)

    start = non_zero[0]
    end = start
    for idx in non_zero[1:]:
        if idx == end + 1:
            end = idx
        else:
            break

    return int(start), int(end)


def _extract_raw_span_from_offsets(raw_text, offsets, a, b):
    """
    Why (score): slice raw text using offsets to preserve exact punctuation/spacing,
    aligning better to word-level Jaccard than token decoding.
    """
    if a > b:
        return _clean_spaces(raw_text)

    start_char, end_char = None, None
    for i in range(a, b + 1):
        if i < 0 or i >= len(offsets):
            continue
        s, e = int(offsets[i][0]), int(offsets[i][1])
        if s == 0 and e == 0:
            continue
        start_char = s
        break

    for i in range(b, a - 1, -1):
        if i < 0 or i >= len(offsets):
            continue
        s, e = int(offsets[i][0]), int(offsets[i][1])
        if s == 0 and e == 0:
            continue
        end_char = e
        break

    if start_char is None or end_char is None or end_char <= start_char:
        return _clean_spaces(raw_text)

    return _clean_spaces(str(raw_text)[start_char:end_char])


def _select_best_span_joint(start_probs, end_probs, tw_s, tw_e, max_answer_len=30):
    """
    Why (score): keep same semantics (maximize start/end) but do joint search
    and forbid illegal spans.
    """
    tw_s = int(tw_s)
    tw_e = int(tw_e)
    if tw_e < tw_s:
        return tw_s, tw_s

    s = start_probs[tw_s : tw_e + 1]
    e = end_probs[tw_s : tw_e + 1]

    eps = 1e-12
    s_log = np.log(np.clip(s, eps, 1.0))
    e_log = np.log(np.clip(e, eps, 1.0))
    score = s_log[:, None] + e_log[None, :]

    L = score.shape[0]
    for i in range(L):
        score[i, :i] = -1e9
        j_max = min(L, i + max_answer_len)
        if j_max < L:
            score[i, j_max:] = -1e9

    flat = int(score.argmax())
    a_rel, b_rel = divmod(flat, L)
    return tw_s + int(a_rel), tw_s + int(b_rel)


final_output = []

if HAS_LOCAL_MODEL:
    raw_texts = df_test["text"].astype(str).values
    raw_sentiments = df_test["sentiment"].astype(str).values

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        base_row = 0
        for _, d in enumerate(tk0):
            bs = d["ids"].shape[0]

            ids = d["ids"].to(device, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)

            outputs_start1, outputs_end1 = model1(
                ids=ids, mask=mask, token_type_ids=token_type_ids
            )
            outputs_start2, outputs_end2 = model2(
                ids=ids, mask=mask, token_type_ids=token_type_ids
            )
            outputs_start3, outputs_end3 = model3(
                ids=ids, mask=mask, token_type_ids=token_type_ids
            )
            outputs_start4, outputs_end4 = model4(
                ids=ids, mask=mask, token_type_ids=token_type_ids
            )
            outputs_start5, outputs_end5 = model5(
                ids=ids, mask=mask, token_type_ids=token_type_ids
            )
            outputs_start6, outputs_end6 = model6(
                ids=ids, mask=mask, token_type_ids=token_type_ids
            )
            outputs_start7, outputs_end7 = model7(
                ids=ids, mask=mask, token_type_ids=token_type_ids
            )
            outputs_start8, outputs_end8 = model8(
                ids=ids, mask=mask, token_type_ids=token_type_ids
            )

            outputs_start = (
                outputs_start1
                + outputs_start2
                + outputs_start3
                + outputs_start4
                + outputs_start5
                + outputs_start7
                + outputs_start6
                + outputs_start8
            ) / 8.0

            outputs_end = (
                outputs_end1
                + outputs_end2
                + outputs_end3
                + outputs_end4
                + outputs_end5
                + outputs_end6
                + outputs_end7
                + outputs_end8
            ) / 8.0

            batch_offsets = d["offsets"].cpu().numpy()
            batch_mask = d["mask"].cpu().numpy()

            off_nonzero = np.zeros((bs, MAX_LEN), dtype=np.float32)
            for i in range(bs):
                valid_len = int(batch_mask[i].sum())
                for j in range(valid_len):
                    s, e = int(batch_offsets[i][j][0]), int(batch_offsets[i][j][1])
                    if not (s == 0 and e == 0):
                        off_nonzero[i, j] = 1.0

            off_nonzero_t = torch.tensor(off_nonzero, device=device)
            very_neg = torch.tensor(-1e9, device=device)
            outputs_start = torch.where(off_nonzero_t > 0.0, outputs_start, very_neg)
            outputs_end = torch.where(off_nonzero_t > 0.0, outputs_end, very_neg)

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().numpy()

            for row_i in range(bs):
                raw_text = raw_texts[base_row + row_i]
                raw_sent = str(raw_sentiments[base_row + row_i]).strip().lower()

                mask_row = batch_mask[row_i].tolist()
                offsets_row = batch_offsets[row_i].tolist()

                tw_s, tw_e = _get_tweet_token_bounds_from_offsets(offsets_row, mask_row)

                if raw_sent == "neutral" or len(_clean_spaces(raw_text).split()) <= 2:
                    selected_text = raw_text
                else:
                    a, b = _select_best_span_joint(
                        outputs_start[row_i],
                        outputs_end[row_i],
                        tw_s=tw_s,
                        tw_e=tw_e,
                        max_answer_len=30,
                    )

                    conf = float(outputs_start[row_i][a] * outputs_end[row_i][b])
                    if conf < 1e-6:
                        selected_text = raw_text
                    else:
                        if a < tw_s or b > tw_e:
                            selected_text = raw_text
                        else:
                            selected_text = _extract_raw_span_from_offsets(
                                raw_text=raw_text, offsets=offsets_row, a=a, b=b
                            )
                            if selected_text.strip() == "":
                                selected_text = raw_text

                final_output.append(_clean_spaces(selected_text))

            base_row += bs
else:
    for t, s in zip(df_test["text"].values, df_test["sentiment"].values):
        final_output.append(_best_span_heuristic(t, s))

print("Generated predictions:", len(final_output))



## === cell 8
sub = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

if len(final_output) != len(sub):
    raise RuntimeError(
        f"Prediction length mismatch: {len(final_output)} vs sample_submission {len(sub)}"
    )

sub.loc[:, "selected_text"] = final_output
sub["selected_text"] = sub["selected_text"].fillna("").astype(str)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 9
sub.sample(20, random_state=0)
