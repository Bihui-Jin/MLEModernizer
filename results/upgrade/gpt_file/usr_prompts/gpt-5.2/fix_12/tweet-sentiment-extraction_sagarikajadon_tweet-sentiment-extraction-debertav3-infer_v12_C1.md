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

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scipy==1.15.3
seaborn==0.12.2
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

0.717913031578064

# 6. Current score

0.61101

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.26939) has done: 'I fix the environment/runtime crash caused by an incompatible protobuf version by avoiding unnecessary imports (notably `pandas_profiling`) and using safe, minimal imports. Then I remove hard-coded Kaggle Dataset paths that don’t exist in your environment and instead load the tokenizer/model directly from `CFG.MODEL_NAME` with `local_files_only=True` fallback to online if available. Finally, I make the inference loop robust (handle missing checkpoints, ensure tensors exist, correct softmax dimension, and always generate `final_outputs` of the right length) and write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.01047) has done: 'I fix the runtime crash coming from a protobuf/transformers import incompatibility by avoiding the code path that triggers it and by switching to a locally-available backbone that doesn’t rely on the problematic dependency chain. Then I fix a scoring-critical bug in post-processing: applying softmax over the wrong dimension (it must be over the token dimension), which currently makes the span selection essentially random and explains the very low Jaccard score. Finally, I correct the checkpoint directory to a real path in this environment (or gracefully fall back), keep the core span-extraction logic intact, and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.4734) has done: 'I fix the crash by removing the `torch.hub` fairseq dependency (it requires `hydra-core`, which isn’t installed) and instead load the same `roberta-base` backbone via `transformers`, keeping the exact same “backbone → dropout ensemble → linear head → start/end logits” core logic. I also fix the DataLoader collation issue by returning `text_tokens` as a list of tokens (so batching works), and keep the softmax applied over the token dimension (sequence length). Finally, I ensure inference always runs end-to-end on CPU/GPU and writes a valid `submission.csv` with `textID,selected_text`.'
- What this solution (achieved 0.19336) has done: 'I fix the `transformers`/protobuf crash by forcing the pure-Python protobuf implementation before importing `transformers`, which avoids the `MessageFactory.GetPrototype` failure in this environment. Then I fix the batching bug that makes `text_tokens` length not match the number of rows: the default DataLoader collate “transposes” lists of tokens across the batch; I provide a minimal custom `collate_fn` that keeps `text_tokens` and original fields aligned per sample. Finally, I keep the same span-selection logic but ensure the model has actual learned weights by loading Kaggle’s provided fold checkpoints if present (otherwise it still run with random weights), and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.22153) has done: 'You’re crashing at `transformers` import because `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` is being set too late; it must be set before any `transformers`/protobuf-related import to avoid the `MessageFactory.GetPrototype` error in this environment. I also fix the test file paths to use the provided `/kaggle/input/...` layout so the notebook runs end-to-end without relying on non-existent `/kaggle/data` paths. To move the score toward your target, I add a minimal and competition-standard post-processing step: instead of taking independent argmax start/end, we choose the best (start,end) span by maximizing `start_prob * end_prob` under a max-span-length constraint and ignoring special tokens; this keeps the same model and logits but improves span selection significantly. The script still always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.20987) has done: 'I fix the crash by replacing the missing local RoBERTa BPE-file tokenizer with a `transformers` RobertaTokenizerFast loaded from `CFG.MODEL_NAME` (using `local_files_only=True` fallback), which restores `CFG.TOKENIZER` so downstream cells run. I keep your GRU QA model and span-selection logic intact, only adjusting special-token handling to use the tokenizer’s actual special ids/tokens and ensuring masking/softmax are applied safely. I also make checkpoint discovery deterministic and non-blocking, and guarantee a correctly formatted `submission.csv` is written even if no checkpoint is found (score be low but valid). These changes are execution-critical and score-positive relative to “not yielded” because they enable end-to-end inference and correct tokenization.'
- What this solution (achieved 0.1886) has done: 'Your current score is low mainly because the model is effectively untrained: it uses a randomly-initialized embedding+GRU and usually finds no compatible checkpoints, so span logits are near-random and the span selection can’t recover. To move toward the 0.7179 target with minimal core-logic change, I keep your exact GRU QA head and inference pipeline, but initialize the embedding from the same RoBERTa tokenizer’s word embedding matrix (via `transformers.RobertaModel`) and freeze it; this preserves your architecture while giving meaningful token representations. I also ensure the selected span is mapped back to the *original tweet text* using offset mappings (still using your start/end selection), which is critical for word-level Jaccard because token-string reconstruction often mismatches whitespace/punctuation. These changes are small, metric-aligned, and should substantially raise the score without changing the training loop (you aren’t training here) or the span-selection semantics.'
- What this solution (achieved 0.61101) has done: 'I fix the runtime crash caused by the `protobuf`/`transformers` incompatibility by avoiding `transformers.RobertaModel` entirely (that call triggers the `MessageFactory.GetPrototype` error), while keeping your GRU QA head and inference/span-selection logic unchanged. To move the score upward toward your target, I replace the currently-untrained (random) model logits with a minimal, competition-standard fallback when no checkpoint is found: a deterministic lexicon/neutral heuristic selection that generally scores far higher than random spans on this dataset. If a checkpoint is found and loads successfully, the model-based predictions are kept as-is (so we don’t change semantics when trained weights exist). The script still run end-to-end and always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import re
import random
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
class CFG:
    DEBUG = False
    TRAIN = True
    N_FOLDS = 5
    TRAIN_FOLDS = [i for i in range(N_FOLDS)]
    SEED = 42
    TEST_BATCHSIZE = 100
    MAX_LENGTH = 128

    MODEL_NAME = "roberta-base"

    FC_DROPOUT = [0.1, 0.2, 0.3, 0.4, 0.5]




## === cell 2
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed=CFG.SEED)




## === cell 3
TEST_PATH = "/kaggle/input/tweet-sentiment-extraction/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
TRAIN_PATH = "/kaggle/input/tweet-sentiment-extraction/train.csv"

test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
train_df = pd.read_csv(TRAIN_PATH)

test_df.head()




## === cell 4
from transformers import RobertaTokenizerFast


def load_roberta_tokenizer(model_name: str):
    try:
        tok = RobertaTokenizerFast.from_pretrained(model_name, local_files_only=True)
        return tok
    except Exception as e_local:
        try:
            tok = RobertaTokenizerFast.from_pretrained(model_name)
            return tok
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer for {model_name} (local_files_only and online fallback both failed). "
                f"local error={type(e_local).__name__}: {e_local} ; fallback error={type(e).__name__}: {e}"
            )


CFG.TOKENIZER = load_roberta_tokenizer(CFG.MODEL_NAME)

CFG.PAD_ID = int(CFG.TOKENIZER.pad_token_id)
CFG.BOS_ID = int(
    CFG.TOKENIZER.bos_token_id
    if CFG.TOKENIZER.bos_token_id is not None
    else CFG.TOKENIZER.cls_token_id
)
CFG.EOS_ID = int(
    CFG.TOKENIZER.eos_token_id
    if CFG.TOKENIZER.eos_token_id is not None
    else CFG.TOKENIZER.sep_token_id
)




## === cell 5
class QADataset:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = " ".join(str(self.df.text.iloc[item]).split())
        sentiment = str(self.df.sentiment.iloc[item])
        input_text = text + " </s> " + sentiment

        enc = CFG.TOKENIZER(
            input_text,
            add_special_tokens=True,
            max_length=CFG.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        ids = enc["input_ids"].squeeze(0).long()
        attention_mask = enc["attention_mask"].squeeze(0).long()
        offsets = (
            enc["offset_mapping"].squeeze(0).long()
        )  # [T,2] char offsets into input_text

        tok_text_tokens = CFG.TOKENIZER.convert_ids_to_tokens(ids.tolist())

        return {
            "input_ids": ids,
            "mask": attention_mask,
            "offsets": offsets,
            "text_tokens": tok_text_tokens,  # list[str] length MAX_LENGTH
            "orig_text": self.df.text.iloc[item],  # original text for exact output
            "orig_sentiment": self.df.sentiment.iloc[item],
            "norm_text": text,  # normalized text used inside input_text
        }




## === cell 6
class QAModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pretrained = pretrained

        vocab_size = int(getattr(CFG.TOKENIZER, "vocab_size", len(CFG.TOKENIZER)))
        hidden = 192

        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=CFG.PAD_ID)

        self.emb_proj = None

        self.encoder = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_dropout = nn.ModuleList([nn.Dropout(val) for val in CFG.FC_DROPOUT])
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, input_ids, mask, token_type_ids=None):
        x = self.embedding(input_ids)  # [B,T,H]
        x, _ = self.encoder(x)  # [B,T,2H]

        logits = None
        for dropout_layer in self.fc_dropout:
            cur = self.fc(dropout_layer(x))  # [B,T,2]
            logits = cur if logits is None else (logits + cur)

        logits = logits / len(CFG.FC_DROPOUT)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        large_neg = torch.finfo(start_logits.dtype).min / 2
        start_logits = start_logits.masked_fill(mask == 0, large_neg)
        end_logits = end_logits.masked_fill(mask == 0, large_neg)
        return start_logits, end_logits




## === cell 7
def qa_collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    mask = torch.stack([b["mask"] for b in batch], dim=0)
    offsets = torch.stack([b["offsets"] for b in batch], dim=0)
    text_tokens = [b["text_tokens"] for b in batch]
    orig_text = [b["orig_text"] for b in batch]
    orig_sentiment = [b["orig_sentiment"] for b in batch]
    norm_text = [b["norm_text"] for b in batch]
    return {
        "input_ids": input_ids,
        "mask": mask,
        "offsets": offsets,
        "text_tokens": text_tokens,
        "orig_text": orig_text,
        "orig_sentiment": orig_sentiment,
        "norm_text": norm_text,
    }


def test_fn(dataloader, model):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_mask = []
    fin_offsets = []
    fin_text_tokens = []
    fin_orig_text = []
    fin_orig_sentiment = []
    fin_norm_text = []

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["mask"].to(device)

            start_logits, end_logits = model(input_ids, mask)

            fin_output_start.append(start_logits.detach().cpu())
            fin_output_end.append(end_logits.detach().cpu())
            fin_mask.append(mask.detach().cpu())
            fin_offsets.append(data["offsets"].detach().cpu())

            fin_text_tokens.extend(data["text_tokens"])
            fin_orig_text.extend(data["orig_text"])
            fin_orig_sentiment.extend(data["orig_sentiment"])
            fin_norm_text.extend(data["norm_text"])

    fin_output_start = torch.vstack(fin_output_start)
    fin_output_end = torch.vstack(fin_output_end)
    fin_mask = torch.vstack(fin_mask)
    fin_offsets = torch.vstack(fin_offsets)  # [N,T,2]

    return (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_offsets,
        fin_text_tokens,
        fin_orig_text,
        fin_orig_sentiment,
        fin_norm_text,
    )




## === cell 8
def try_load_checkpoint(model, ckpt_path):
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        return False
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        kk = k
        if kk.startswith("model."):
            kk = kk[len("model.") :]
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        new_state[kk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(
        f"Loaded checkpoint: {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return True


def find_possible_checkpoints():
    candidates = []
    bases = ["/kaggle/working", "/kaggle/input/tweet-sentiment-extraction"]
    for base in bases:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                lf = f.lower()
                if lf.endswith((".bin", ".pth", ".pt")) and (
                    ("fold" in lf) or ("model" in lf) or ("checkpoint" in lf)
                ):
                    candidates.append(os.path.join(root, f))
    return sorted(set(candidates))


test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    collate_fn=qa_collate_fn,
)

model = QAModel(pretrained=False).to(device)

loaded = False
for ckpt in find_possible_checkpoints():
    if try_load_checkpoint(model, ckpt):
        loaded = True
        break
if not loaded:
    print(
        "No checkpoint found; will use heuristic-only predictions (score-positive vs random untrained logits)."
    )

if loaded:
    (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_offsets,
        fin_text_tokens,
        fin_orig_text,
        fin_orig_sentiment,
        fin_norm_text,
    ) = test_fn(test_loader, model)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




## === cell 9
if loaded:
    fin_output_start = torch.softmax(fin_output_start, dim=1)
    fin_output_end = torch.softmax(fin_output_end, dim=1)




## === cell 10
positive_words = set(
    [
        "good",
        "great",
        "love",
        "like",
        "awesome",
        "best",
        "amazing",
        "happy",
        "nice",
        "fantastic",
        "wonderful",
        "cool",
        "yay",
        "thanks",
        "thank",
        "excellent",
        "beautiful",
    ]
)
negative_words = set(
    [
        "bad",
        "hate",
        "terrible",
        "awful",
        "worst",
        "sad",
        "angry",
        "mad",
        "sucks",
        "suck",
        "disappointed",
        "pain",
        "annoying",
        "ugh",
        "sorry",
        "fail",
        "horrible",
    ]
)


def clean_spaces(s: str) -> str:
    return " ".join(str(s).split()).strip()


def heuristic_select(text: str, sentiment: str) -> str:
    text0 = str(text)
    s = str(sentiment).lower().strip()
    if s == "neutral":
        return clean_spaces(text0)

    words = re.findall(r"\w+|[^\w\s]", text0, flags=re.UNICODE)
    if not words:
        return clean_spaces(text0)

    lex = positive_words if s == "positive" else negative_words
    hits = [i for i, w in enumerate(words) if w.lower() in lex]
    if not hits:
        return clean_spaces(text0)

    i0, i1 = min(hits), max(hits)
    sel = "".join(
        [
            w if re.match(r"[^\w\s]", w) else ((" " + w) if k > i0 else w)
            for k, w in enumerate(words[i0 : i1 + 1])
        ]
    ).strip()
    return clean_spaces(sel) if sel else clean_spaces(text0)


final_outputs = []

if not loaded:
    for t, s in zip(test_df["text"].values, test_df["sentiment"].values):
        final_outputs.append(heuristic_select(t, s))
else:
    n = fin_output_start.shape[0]
    assert n == fin_output_end.shape[0] == fin_mask.shape[0], (
        fin_output_start.shape,
        fin_output_end.shape,
        fin_mask.shape,
    )
    assert len(fin_text_tokens) == n, (len(fin_text_tokens), n)
    assert len(fin_orig_text) == n, (len(fin_orig_text), n)

    special_set = set()
    for tok in [
        getattr(CFG.TOKENIZER, "cls_token", None),
        getattr(CFG.TOKENIZER, "sep_token", None),
        getattr(CFG.TOKENIZER, "pad_token", None),
        getattr(CFG.TOKENIZER, "bos_token", None),
        getattr(CFG.TOKENIZER, "eos_token", None),
    ]:
        if tok is not None:
            special_set.add(tok)
    special_set.update(["positive", "negative", "neutral"])
    special_set = {x for x in special_set if x is not None}

    MAX_ANSWER_LEN = 30

    for j in range(n):
        toks = fin_text_tokens[j]
        mask = fin_mask[j].numpy().astype(np.float32)

        start_probs = fin_output_start[j].numpy() * mask
        end_probs = fin_output_end[j].numpy() * mask

        valid = np.ones_like(mask, dtype=np.float32)
        for i, tk in enumerate(toks):
            if tk in special_set:
                valid[i] = 0.0

        start_probs = start_probs * valid
        end_probs = end_probs * valid

        T = len(start_probs)
        best_s, best_e = 0, 0
        best_score = -1.0

        for s in range(T):
            if start_probs[s] <= 0:
                continue
            e_max = min(T - 1, s + MAX_ANSWER_LEN)
            e = int(s + np.argmax(end_probs[s : e_max + 1]))
            score = float(start_probs[s] * end_probs[e])
            if score > best_score:
                best_score = score
                best_s, best_e = s, e

        if best_e < best_s:
            best_e = best_s

        offsets = fin_offsets[
            j
        ].numpy()  # [T,2] into input_text = norm_text + " </s> " + sentiment
        norm_text = str(fin_norm_text[j])
        tweet_len = len(norm_text)

        char_starts = []
        char_ends = []
        for k in range(best_s, best_e + 1):
            a, b = int(offsets[k, 0]), int(offsets[k, 1])
            if a == 0 and b == 0:
                continue
            if a >= tweet_len:
                continue
            b = min(b, tweet_len)
            if b > a:
                char_starts.append(a)
                char_ends.append(b)

        if char_starts and char_ends:
            cs, ce = min(char_starts), max(char_ends)
            final_output = norm_text[cs:ce].strip()
            final_output = clean_spaces(final_output)
        else:
            output_tokens = toks[best_s : best_e + 1]
            output_tokens = [
                x for x in output_tokens if x not in special_set and x is not None
            ]
            final_output = CFG.TOKENIZER.convert_tokens_to_string(output_tokens).strip()
            final_output = clean_spaces(final_output)

        if final_output == "" or len(final_output) <= 1:
            final_output = heuristic_select(fin_orig_text[j], fin_orig_sentiment[j])

        final_outputs.append(final_output)




## === cell 11
assert len(final_outputs) == len(test_df), (len(final_outputs), len(test_df))

sub = pd.DataFrame({"textID": test_df["textID"].values, "selected_text": final_outputs})
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
