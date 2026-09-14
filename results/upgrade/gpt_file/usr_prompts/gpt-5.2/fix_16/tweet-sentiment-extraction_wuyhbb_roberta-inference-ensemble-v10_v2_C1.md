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

0.7257987260818481

# 6. Current score

0.50088

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fix the offline model/tokenizer loading so the notebook can run in Kaggle without requiring any extra attached datasets. Specifically, I switch to using a standard Hugging Face RoBERTa model (`roberta-base`) with `local_files_only=True` (and allow download if Kaggle has internet enabled), and I remove the hard dependency on the missing `roberta-weights-v10` fold checkpoints by falling back to a deterministic baseline prediction when weights aren’t present. This guarantees a valid `submission.csv` is always produced end-to-end (unblocking scoring), while keeping your existing dataset/tokenization/prediction core logic intact where possible. If the weights are available, the original ensemble inference path is used unchanged.'
- What this solution (achieved 0.59324) has done: 'The crash comes from a known protobuf incompatibility that can surface when importing/using `transformers` in some Kaggle images; we fix it by forcing the pure-Python protobuf implementation via an environment variable *before* any transformer-related import happens. I also make the model/tokenizer loading more robust in offline mode (keep the same RoBERTa backbone and inference logic) and ensure we always produce a valid `submission.csv` even if weights are missing. These changes are score-neutral when weights exist (your original ensemble path remains), and unblock execution so you can submit; if weights are absent, the baseline fallback remains as before.'
- What this solution (achieved 0.59324) has done: 'You’re hitting a protobuf runtime incompatibility that can still surface even after setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`; the safest minimal fix in this Kaggle image is to also disable the C++ implementation explicitly before importing `transformers`. I add `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` early, and make tokenizer/model loading always use `local_files_only=True` first to avoid triggering problematic downloads/import paths. I also guard the `transformers` import with this env setup (without changing your model/inference logic), so the notebook runs end-to-end and writes a valid `submission.csv`. No score-tuning changes are introduced; once it runs, your score should improve automatically if the fold weights exist, otherwise it still produce a valid baseline submission.'
- What this solution (achieved 0.59324) has done: 'The crash happens before inference completes due to a protobuf/transformers incompatibility triggered when loading Hugging Face models/tokenizers in this Kaggle image. I fix this by enforcing the pure-Python protobuf runtime and disabling the C++ implementation *before* any `transformers` import, which avoids the `MessageFactory.GetPrototype` error. I also make model/tokenizer loading strictly offline-first and, if any transformers-related load still fails at runtime, fall back to the existing baseline submission so a valid `submission.csv` is always produced. These changes are execution/stability fixes and do not alter your core model/inference logic when weights and model assets are available.'
- What this solution (achieved 0.59324) has done: 'We fix the protobuf/transformers crash by forcing a compatible protobuf runtime *before* any `transformers` import and by proactively switching to the pure-Python protobuf implementation in-process. Then we make the `transformers` import and model/tokenizer loading robust: if it still fails, we fall back cleanly to the baseline (full tweet text) so a valid `submission.csv` is always produced. This is a stability fix that unblocks running end-to-end; if the fold weights/model assets exist, your original ensemble inference path remains unchanged and should raise score toward the target. Finally, we also ensure `word_preds` always has the correct length and no NaNs before writing the submission.'
- What this solution (achieved 0.59324) has done: 'We fix the protobuf incompatibility that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by forcing a compatible protobuf version before importing `transformers` (this error is commonly caused by protobuf 4+/5+ with some transformer/tokenizer codepaths). The change is minimal: add an early pip install for `protobuf==3.20.3` (fast, small) and then import `transformers` after that, keeping your existing model/dataset/inference logic intact. This should allow the RoBERTa tokenizer/model to load successfully; if the fold weights exist, your original ensemble path run and should improve score toward the target (vs the baseline full-text fallback). We also keep the baseline fallback and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.48997) has done: 'Your current 0.59324 score is consistent with the fallback path (predicting the full tweet text), which usually happens because the external fold checkpoints (`roberta-weights-v10`) are not available. To move the score upward toward the 0.7258 target without changing core model logic, I (1) stop depending on the missing external weights by training the existing `TweetModel` for a very short, fixed number of steps on `train.csv` using the same start/end extraction objective implied by your inference, then (2) run the same inference code path on test. I also add the minimal label construction (token-level start/end indices via offsets) and a small train/valid split only for monitoring (no early stopping), keeping everything deterministic and within the time budget. This should raise the score substantially above the baseline while preserving your architecture and prediction decoding semantics.'
- What this solution (achieved 0.54472) has done: 'You’re far below the 0.7258 target (0.48997), so we should improve score with minimal, metric-aligned fixes while keeping your model/training loop and decoding intact. The biggest low-risk gain here is correcting the token-index alignment bug: your dataset constructs labels in “raw token indices” but inference subtracts `args.offset` (4), so training is teaching a different indexing scheme than decoding uses, hurting Jaccard substantially. I also clamp start/end labels to valid positions and ensure neutral examples don’t corrupt span learning (still no change to your neutral-at-inference rule). These changes preserve your architecture, loss, and training approach, but make training/inference consistent so the on-the-fly trained model can move score toward the target.'
- What this solution (achieved 0.53955) has done: 'Your score gap to the target is still large (0.54472 vs 0.7258; higher is better), so the smallest safe way to move up is to improve training/label alignment without changing your model or decoding semantics. I keep the same TweetModel, same max_len, same loss (start/end CE), and the same neutral-at-inference rule, but fix two training-time issues that typically cap Jaccard: (1) align label creation to the exact token offsets used by the “text” sequence (ignoring the sentiment/special tokens) and (2) handle cases where `selected_text` occurs multiple times by choosing the occurrence that best matches token span coverage. These are minimal, metric-aligned changes that should improve span learning and therefore selected_text quality while keeping runtime within the existing step budget and producing the same submission format.'
- What this solution (achieved 0.61018) has done: 'We’re still well below the target (0.53955 vs 0.7258; higher is better), so we should improve span quality with the smallest changes that keep your model/training/decoding intact. The biggest likely limiter now is that RoBERTa pair-encoding offsets don’t map cleanly to the original tweet string, so your label spans and decoding can be misaligned; we fix this by tokenizing a single sequence that explicitly contains the sentiment (as text) plus a separator plus the tweet, so the returned offsets always refer to the exact same string used for label-building and for decoding. We keep the same TweetModel, same start/end CE loss, same training loop, and the same “neutral => full text” rule; only the dataset text/offset construction changes and we adjust the `args.offset` accordingly (still a constant). This should legitimately improve training signal and extraction alignment, moving the score upward toward the target band while staying within time and producing the same `submission.csv` format.'
- What this solution (achieved 0.54129) has done: 'Your current score (0.61018) is well below the target (0.7258), so we should make a small, metric-aligned fix that improves extraction quality without changing your model, loss, or training loop. The biggest issue is that `args.offset` is set to 0 while the dataset still adds start/end labels with an extra offset, so training and decoding are misaligned; we compute the correct offset directly from the tokenizer’s produced offsets for this exact “sentiment + sep + tweet” encoding and use it consistently everywhere. This keeps the same architecture and decoding semantics, but makes the start/end indices refer to the same token region during both training and inference. Additionally, we constrain start/end predictions to the tweet-token region (non-zero offsets) so the model can’t select sentiment/separator special tokens, which typically improves Jaccard with minimal risk.'
- What this solution (achieved 0.50088) has done: 'I fix the remaining training/inference misalignment that is still limiting your span model: the dataset currently *adds* `offset` to the start/end labels even though your offsets are already shifted into “tweet-only token space”, and inference *subtracts* `args.offset`, so the model is trained to point at the wrong indices. I also make the masking consistent during training by preventing the loss from ever learning to predict non-tweet tokens (those with offset `(0,0)`), matching what you already do at inference time. These are minimal, metric-aligned changes (no architecture/loss/training-loop redesign) that should improve Jaccard materially vs the current fallback-like behavior and move the score toward your 0.7258 target. The script still runs end-to-end and always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".", 1)[0])
        if major >= 4:
            raise RuntimeError(f"protobuf too new: {pb_ver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                sys.modules.pop(m, None)


_ensure_protobuf_compat()

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C", "1")

try:
    import google.protobuf.internal.api_implementation as _api_impl

    try:
        _api_impl._SetImplementationType("python")
    except Exception:
        pass
except Exception:
    pass

import re
import random
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    RobertaTokenizerFast,
    AutoConfig,
    AutoModel,
)




## === cell 1
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model: nn.Module, path: str, map_location: str = "cpu") -> None:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if len(unexpected) > 50:
        print("Warning: many unexpected keys while loading:", len(unexpected))


def ensemble(
    all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, df, softmax=True
):
    whole = np.mean(np.stack(all_whole_preds, axis=0), axis=0)

    n = len(df)
    start_list, end_list, inst_list = [], [], []
    for i in range(n):
        s = torch.stack(
            [all_start_preds[f][i].float() for f in range(len(all_start_preds))], dim=0
        ).mean(0)
        e = torch.stack(
            [all_end_preds[f][i].float() for f in range(len(all_end_preds))], dim=0
        ).mean(0)
        ins = torch.stack(
            [all_inst_preds[f][i].float() for f in range(len(all_inst_preds))], dim=0
        ).mean(0)
        start_list.append(s)
        end_list.append(e)
        inst_list.append(ins)
    return whole, start_list, end_list, inst_list


def _clean_text_basic(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return " ".join(str(x).split())


def _select_from_offsets(
    text: str, offsets: List[Tuple[int, int]], start_idx: int, end_idx: int
) -> str:
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    start_idx = max(0, min(start_idx, len(offsets) - 1))
    end_idx = max(0, min(end_idx, len(offsets) - 1))

    spans = [
        (s, e)
        for (s, e) in offsets[start_idx : end_idx + 1]
        if not (s == 0 and e == 0) and e >= s
    ]
    if not spans:
        return _clean_text_basic(text)
    s_char = min(s for s, _ in spans)
    e_char = max(e for _, e in spans)
    if e_char <= s_char:
        return _clean_text_basic(text)
    return text[s_char:e_char]


def get_predicts_from_token_logits(
    whole_pred: np.ndarray,
    start_pred: List[torch.Tensor],
    end_pred: List[torch.Tensor],
    inst_pred: List[torch.Tensor],
    df: pd.DataFrame,
    args,
):
    word_preds, inst_word_preds, scores = [], [], []
    for i, row in df.iterrows():
        text = str(row["text"])
        sentiment = str(row["sentiment"])

        if sentiment == "neutral":
            pred = _clean_text_basic(text)
            word_preds.append(pred)
            inst_word_preds.append(pred)
            scores.append(0.0)
            continue

        s_logits = start_pred[i]
        e_logits = end_pred[i]

        s_idx = int(torch.argmax(s_logits).item())
        e_idx = int(torch.argmax(e_logits).item())

        offsets = df.at[i, "offsets"]
        pred = _select_from_offsets(
            text, offsets, s_idx - args.offset, e_idx - args.offset
        )
        pred = _clean_text_basic(pred)

        word_preds.append(pred)
        inst_word_preds.append(pred)
        scores.append(float(whole_pred[i]) if whole_pred is not None else 0.0)

    return word_preds, inst_word_preds, scores


def _char_span_to_token_span(offsets: List[Tuple[int, int]], s_char: int, e_char: int):
    cand = []
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if e > s_char and s < e_char:
            cand.append(i)
    if not cand:
        return None, None
    return cand[0], cand[-1]


def _all_occurrences(text: str, sub: str) -> List[int]:
    if sub == "":
        return []
    starts = []
    i = 0
    while True:
        j = text.find(sub, i)
        if j == -1:
            break
        starts.append(j)
        i = j + 1
    return starts


def _best_occurrence_token_span(
    text: str, selected: str, offsets: List[Tuple[int, int]]
) -> Tuple[int, int]:
    starts = _all_occurrences(text, selected)
    if not starts:
        return None, None

    best = None
    best_err = None
    for s_char in starts:
        e_char = s_char + len(selected)
        s_tok, e_tok = _char_span_to_token_span(offsets, s_char, e_char)
        if s_tok is None:
            continue
        recon = _select_from_offsets(text, offsets, s_tok, e_tok)
        err = abs(len(_clean_text_basic(recon)) - len(_clean_text_basic(selected)))
        if best is None or err < best_err:
            best = (s_tok, e_tok)
            best_err = err

    if best is None:
        return None, None
    return best




## === cell 2
class TrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        labels=None,
        tokenizer=None,
        mode="train",
        offset=4,
        max_len=96,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode
        self.offset = offset
        self.max_len = max_len

        self.encodings = []
        self.offsets = []

        self.sep = " </s> "
        for _, row in self.df.iterrows():
            text = _clean_text_basic(row["text"])
            sentiment = str(row["sentiment"])

            combined = f"{sentiment}{self.sep}{text}"
            enc = self.tokenizer(
                combined,
                add_special_tokens=True,
                return_offsets_mapping=True,
                padding=False,
                truncation=True,
                max_length=self.max_len,
            )

            off = enc["offset_mapping"]
            tweet_start_char = combined.find(self.sep) + len(self.sep)
            new_off = []
            for s, e in off:
                if s >= tweet_start_char and e >= tweet_start_char and e > s:
                    new_off.append((s - tweet_start_char, e - tweet_start_char))
                else:
                    new_off.append((0, 0))

            enc["offset_mapping"] = new_off
            self.encodings.append(enc)
            self.offsets.append(new_off)

        self.df["offsets"] = self.offsets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(
            enc.get("token_type_ids", [0] * len(enc["input_ids"])), dtype=torch.long
        )
        dummy = torch.tensor(0, dtype=torch.long)
        return (
            input_ids,
            token_type_ids,
            attention_mask,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
        )


class SpanTrainDataset(Dataset):
    """
    Minimal score-improvement fix:
    - Offsets are already shifted so that ONLY tweet tokens have non-(0,0) offsets.
    - Inference subtracts args.offset, so labels must be stored in the SAME
      token-index space as model outputs (no extra +offset here).
    """

    def __init__(self, df: pd.DataFrame, tokenizer=None, offset=4, max_len=96):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.offset = offset
        self.max_len = max_len

        self.encodings = []
        self.offsets = []
        self.start_positions = []
        self.end_positions = []

        self.sep = " </s> "
        for _, row in self.df.iterrows():
            text = _clean_text_basic(row["text"])
            selected = _clean_text_basic(row["selected_text"])
            sentiment = str(row["sentiment"])

            combined = f"{sentiment}{self.sep}{text}"
            enc = self.tokenizer(
                combined,
                add_special_tokens=True,
                return_offsets_mapping=True,
                padding=False,
                truncation=True,
                max_length=self.max_len,
            )

            off = enc["offset_mapping"]
            tweet_start_char = combined.find(self.sep) + len(self.sep)
            new_off = []
            for s, e in off:
                if s >= tweet_start_char and e >= tweet_start_char and e > s:
                    new_off.append((s - tweet_start_char, e - tweet_start_char))
                else:
                    new_off.append((0, 0))
            enc["offset_mapping"] = new_off

            offsets = new_off
            seq_len = len(enc["input_ids"])

            if sentiment == "neutral" or selected == "":
                s_tok = 0
                e_tok = 0
            else:
                s_tok, e_tok = _best_occurrence_token_span(text, selected, offsets)
                if s_tok is None:
                    s_tok = 0
                    e_tok = 0

            s_tok = max(0, min(int(s_tok), seq_len - 1))
            e_tok = max(0, min(int(e_tok), seq_len - 1))

            self.encodings.append(enc)
            self.offsets.append(offsets)

            self.start_positions.append(s_tok)
            self.end_positions.append(e_tok)

        self.df["offsets"] = self.offsets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(
            enc.get("token_type_ids", [0] * len(enc["input_ids"])), dtype=torch.long
        )
        start_pos = torch.tensor(self.start_positions[idx], dtype=torch.long)
        end_pos = torch.tensor(self.end_positions[idx], dtype=torch.long)
        dummy = torch.tensor(0, dtype=torch.long)
        return (
            input_ids,
            token_type_ids,
            attention_mask,
            start_pos,
            end_pos,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
        )


class MyCollator:
    def __call__(self, batch):
        input_ids, token_type_ids, attention_mask, d1, d2, d3, d4, d5, d6, d7 = zip(
            *batch
        )
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=1
        )  # roberta pad id is 1
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        d1 = torch.stack(d1)
        d2 = torch.stack(d2)
        d3 = torch.stack(d3)
        d4 = torch.stack(d4)
        d5 = torch.stack(d5)
        d6 = torch.stack(d6)
        d7 = torch.stack(d7)

        return (
            input_ids,
            token_type_ids,
            attention_mask,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
        )




## === cell 3
set_seed(42)
DATA_DIR = "../input/tweet-sentiment-extraction"
if not os.path.exists(os.path.join(DATA_DIR, "test.csv")):
    DATA_DIR = "../kaggle/input/tweet-sentiment-extraction"
if not os.path.exists(os.path.join(DATA_DIR, "test.csv")):
    DATA_DIR = "/kaggle/input/tweet-sentiment-extraction"
if not os.path.exists(os.path.join(DATA_DIR, "test.csv")):
    DATA_DIR = "/kaggle/data/tweet-sentiment-extraction"

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))




## === cell 4
def _load_tokenizer_and_config(model_name: str = "roberta-base"):
    try:
        tok = RobertaTokenizerFast.from_pretrained(model_name, local_files_only=True)
        cfg = AutoConfig.from_pretrained(
            model_name, output_hidden_states=True, local_files_only=True
        )
        return tok, cfg, model_name
    except Exception as e1:
        try:
            tok = RobertaTokenizerFast.from_pretrained(model_name)
            cfg = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
            return tok, cfg, model_name
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load tokenizer/config for {model_name}. "
                f"offline_error={repr(e1)} online_error={repr(e2)}"
            )


tokenizer, base_config, PRETRAIN_NAME = _load_tokenizer_and_config("roberta-base")


def _compute_tweet_token_offset(tokenizer, max_len: int = 96) -> int:
    sep = " </s> "
    combined = f"positive{sep}hello"
    enc = tokenizer(
        combined,
        add_special_tokens=True,
        return_offsets_mapping=True,
        padding=False,
        truncation=True,
        max_length=max_len,
    )
    offsets = enc["offset_mapping"]
    tweet_start_char = combined.find(sep) + len(sep)
    for i, (s, e) in enumerate(offsets):
        if s >= tweet_start_char and e > s:
            return int(i)
    return 0


class Args:
    post = False
    tokenizer = tokenizer
    offset = _compute_tweet_token_offset(tokenizer, max_len=96)
    batch_size = 16
    workers = 1
    max_len = 96
    lr = 2e-5
    weight_decay = 0.01
    epochs = 1
    max_train_steps = 600


args = Args()
print("Using computed args.offset (tweet token start index) =", args.offset)

collator = MyCollator()

test_set = TrainDataset(
    test,
    None,
    tokenizer=tokenizer,
    mode="test",
    offset=args.offset,
    max_len=args.max_len,
)
test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collator,
    num_workers=args.workers,
    pin_memory=torch.cuda.is_available(),
)

perm = np.random.RandomState(42).permutation(len(train))
val_size = int(0.05 * len(train))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

train_trn = train.iloc[trn_idx].reset_index(drop=True)
train_val = train.iloc[val_idx].reset_index(drop=True)

train_trn_set = SpanTrainDataset(
    train_trn, tokenizer=tokenizer, offset=args.offset, max_len=args.max_len
)
train_val_set = SpanTrainDataset(
    train_val, tokenizer=tokenizer, offset=args.offset, max_len=args.max_len
)

train_trn_loader = DataLoader(
    train_trn_set,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collator,
    num_workers=args.workers,
    pin_memory=torch.cuda.is_available(),
)
train_val_loader = DataLoader(
    train_val_set,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collator,
    num_workers=args.workers,
    pin_memory=torch.cuda.is_available(),
)




## === cell 5
class TweetModel(nn.Module):

    def __init__(self, pretrain_path=None, dropout=0.2, config=None):
        super(TweetModel, self).__init__()
        if config is not None and pretrain_path is None:
            self.bert = AutoModel.from_config(config)
        else:
            try:
                config = AutoConfig.from_pretrained(
                    pretrain_path, output_hidden_states=True, local_files_only=True
                )
                self.bert = AutoModel.from_pretrained(
                    pretrain_path, cache_dir=None, config=config, local_files_only=True
                )
            except Exception:
                config = AutoConfig.from_pretrained(
                    pretrain_path, output_hidden_states=True
                )
                self.bert = AutoModel.from_pretrained(
                    pretrain_path, cache_dir=None, config=config
                )

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
        out = self.bert(
            inputs,
            attention_mask=masks,
            token_type_ids=token_type_ids,
            inputs_embeds=input_emb,
        )
        if len(out) >= 3:
            last_hidden = out[0]
            pooled_output = out[1] if out[1] is not None else last_hidden[:, 0]
            hs = out[2]
        else:
            last_hidden = out[0]
            pooled_output = last_hidden[:, 0]
            hs = out.hidden_states

        seq_output = torch.cat([hs[-1], hs[-2], hs[-3]], dim=-1)

        avg_output = F.adaptive_avg_pool1d(seq_output.permute(0, 2, 1), 1).squeeze(-1)
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0, 2, 1)).permute(0, 2, 1))

        se_out = self.se_head(self.dropout(seq_output))
        inst_out = self.inst_head(self.dropout(seq_output))
        return whole_out, se_out[:, :, 0], se_out[:, :, 1], inst_out




## === cell 6
def predict(
    model: nn.Module, valid_df, valid_loader, args, device, progress=False
) -> Dict[str, float]:
    model.eval()
    all_end_pred, all_whole_pred, all_start_pred, all_inst_out = [], [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for tokens, types, masks, _, _, _, _, _, _, _ in valid_loader:
            if progress:
                batch_size = tokens.size(0)
                tq.update(batch_size)
            masks = masks.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            types = types.to(device, non_blocking=True)
            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)

            for b in range(tokens.size(0)):
                i_global = len(all_start_pred) + b
                offsets = valid_df.at[i_global, "offsets"]
                valid_tok = torch.tensor(
                    [(s != 0 or e != 0) for (s, e) in offsets],
                    dtype=torch.bool,
                    device=device,
                )
                start_out[b, : valid_tok.numel()] = start_out[
                    b, : valid_tok.numel()
                ].masked_fill(~valid_tok, -1000.0)
                end_out[b, : valid_tok.numel()] = end_out[
                    b, : valid_tok.numel()
                ].masked_fill(~valid_tok, -1000.0)

            start_out = start_out.masked_fill(~masks.bool(), -1000.0)
            end_out = end_out.masked_fill(~masks.bool(), -1000.0)

            start_out = torch.softmax(start_out, dim=-1)
            end_out = torch.softmax(end_out, dim=-1)

            all_whole_pred.append(
                torch.softmax(whole_out, dim=-1)[:, 1].detach().cpu().numpy()
            )
            inst_out = torch.softmax(inst_out, dim=-1)
            for idx in range(len(start_out)):
                all_start_pred.append(start_out[idx, :].detach().cpu())
                all_end_pred.append(end_out[idx, :].detach().cpu())
                all_inst_out.append(inst_out[idx, :, 1].detach().cpu())
            assert all_start_pred[-1].dim() == 1

    all_whole_pred = np.concatenate(all_whole_pred)

    if progress:
        tq.close()
    return all_whole_pred, all_start_pred, all_end_pred, all_inst_out


def _train_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()
    ce = nn.CrossEntropyLoss()
    step = 0
    pbar = tqdm.tqdm(loader, total=min(len(loader), (max_steps or len(loader))))
    for batch in pbar:
        tokens, types, masks, start_pos, end_pos, _, _, _, _, _ = batch
        tokens = tokens.to(device, non_blocking=True)
        types = types.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        start_pos = start_pos.to(device, non_blocking=True)
        end_pos = end_pos.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        _, start_out, end_out, _ = model(tokens, masks, types)

        bs = tokens.size(0)
        for b in range(bs):
            offsets = (
                loader.dataset.df.at[step * loader.batch_size + b, "offsets"]
                if hasattr(loader, "batch_size")
                else loader.dataset.df.at[b, "offsets"]
            )

        start_out = start_out.masked_fill(~masks.bool(), -1000.0)
        end_out = end_out.masked_fill(~masks.bool(), -1000.0)

        loss = ce(start_out, start_pos) + ce(end_out, end_pos)
        loss.backward()
        optimizer.step()

        step += 1
        pbar.set_description(f"train_loss={loss.item():.4f}")
        if max_steps is not None and step >= max_steps:
            break


def _eval_span_acc(model, loader, device, max_batches=50):
    model.eval()
    correct_s = 0
    correct_e = 0
    total = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            tokens, types, masks, start_pos, end_pos, _, _, _, _, _ = batch
            tokens = tokens.to(device, non_blocking=True)
            types = types.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            start_pos = start_pos.to(device, non_blocking=True)
            end_pos = end_pos.to(device, non_blocking=True)

            _, start_out, end_out, _ = model(tokens, masks, types)

            start_out = start_out.masked_fill(~masks.bool(), -1000.0)
            end_out = end_out.masked_fill(~masks.bool(), -1000.0)

            ps = torch.argmax(start_out, dim=-1)
            pe = torch.argmax(end_out, dim=-1)
            correct_s += (ps == start_pos).sum().item()
            correct_e += (pe == end_pos).sum().item()
            total += tokens.size(0)
            if bi + 1 >= max_batches:
                break
    if total == 0:
        return 0.0, 0.0
    return correct_s / total, correct_e / total




## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

word_preds = None

try:
    model = None
    try:
        model = TweetModel(pretrain_path=PRETRAIN_NAME)
    except Exception:
        model = TweetModel(config=base_config)

    model.to(device)

    all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = [], [], [], []

    WEIGHTS_DIR = "../input/roberta-weights-v10"
    if not os.path.exists(WEIGHTS_DIR):
        WEIGHTS_DIR = "../kaggle/input/roberta-weights-v10"
    if not os.path.exists(WEIGHTS_DIR):
        WEIGHTS_DIR = "/kaggle/input/roberta-weights-v10"

    weight_files = [os.path.join(WEIGHTS_DIR, f"best-model-{i}.pt") for i in range(5)]
    have_all_weights = all(os.path.exists(p) for p in weight_files)

    if have_all_weights:
        for fold in range(5):
            load_model(model, weight_files[fold], map_location=device)
            model.to(device)
            fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = (
                predict(
                    model, test_set.df, test_loader, args, device=device, progress=True
                )
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
            test_set.df,
            softmax=True,
        )

        word_preds, inst_word_preds, scores = get_predicts_from_token_logits(
            all_whole_preds,
            all_start_preds,
            all_end_preds,
            all_inst_preds,
            test_set.df,
            args,
        )
    else:
        print(
            f"WARNING: Missing fold weights under {WEIGHTS_DIR}. "
            "Training a minimal on-the-fly model on train.csv to improve score vs baseline."
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        for ep in range(args.epochs):
            _train_one_epoch(
                model,
                train_trn_loader,
                optimizer,
                device,
                max_steps=args.max_train_steps,
            )
            s_acc, e_acc = _eval_span_acc(
                model, train_val_loader, device=device, max_batches=50
            )
            print(f"epoch={ep+1} val_start_acc={s_acc:.3f} val_end_acc={e_acc:.3f}")

        fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = predict(
            model, test_set.df, test_loader, args, device=device, progress=True
        )

        word_preds, inst_word_preds, scores = get_predicts_from_token_logits(
            fold_whole_preds,
            fold_start_preds,
            fold_end_preds,
            fold_inst_preds,
            test_set.df,
            args,
        )

except Exception as e:
    print(
        "ERROR during model/tokenizer inference. Falling back to baseline. Error:",
        repr(e),
    )
    word_preds = [_clean_text_basic(t) for t in test["text"].astype(str).tolist()]

if word_preds is None or len(word_preds) != len(test):
    word_preds = [_clean_text_basic(t) for t in test["text"].astype(str).tolist()]

word_preds = [_clean_text_basic(x) for x in word_preds]



## === cell 8
test["selected_text"] = word_preds
sub = test[["textID", "selected_text"]].copy()
sub.to_csv("submission.csv", index=False)
print(sub.head(10))
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv saved to:", os.path.abspath("submission.csv"))
