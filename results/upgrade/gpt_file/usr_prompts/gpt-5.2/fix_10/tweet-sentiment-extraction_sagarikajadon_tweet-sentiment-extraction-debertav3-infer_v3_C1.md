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

0.2875462770462036

# 6. Current score

0.48811

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.21494) has done: 'I fix the environment/import issue that prevents `transformers` from loading (protobuf / `MessageFactory.GetPrototype`), and make the tokenizer/model loading robust by falling back to the standard `CFG.MODEL_NAME` when the referenced Kaggle input artifacts don’t exist. I also fix missing imports and dataset class definition ordering so `QADataset` and `DataCollatorWithPadding` are available when used. Since your current run never produced a submission, I ensure inference always completes and writes a valid `submission.csv` with the required columns and row count. These changes keep the core DeBERTa QA-style start/end span logic intact while making it runnable end-to-end on the provided dataset paths.'
- What this solution (achieved 0.0) has done: 'I fix the `transformers` import crash by pinning `protobuf` to the pure-Python implementation *and* applying a small compatibility monkey-patch for `google.protobuf.message_factory.MessageFactory.GetPrototype`, which is the root cause of the current runtime error in this environment. Then I correct an inference-time logic issue: you’re thresholding raw logits (not probabilities), which makes the span selection unstable and hurts Jaccard—using `softmax` for start/end probabilities keeps the same QA-span core logic while improving calibration toward your target score. Finally, I ensure DataLoader batching works reliably by using the existing `DataCollatorWithPadding` (or a safe fallback) so tensors are correctly stacked, and keep the submission formatting unchanged.'
- What this solution (achieved 0.48811) has done: 'Your current 0.0 score is consistent with the submission being effectively invalid for the metric because many predictions are empty/garbled: the span selection uses a fixed probability threshold and then chooses the *first* token above threshold for both start and end, and token reconstruction is done from token strings rather than the original character offsets (this often fails for DeBERTa’s SentencePiece tokens and punctuation). To move the score toward your target with minimal change and identical core QA-span logic, I (1) pick start/end by argmax over masked probabilities (still using your start/end distributions, just removing the brittle thresholding), (2) use the tokenizer’s `offset_mapping` to extract the predicted substring from the original tweet text (standard for this competition and aligns with word-level Jaccard), and (3) add the known neutral sentiment shortcut (predict full text) to stabilize results without changing the model. These changes keep the same model, same inference pass, and same submission format, while making predictions non-empty and properly aligned to the tweet text.'
- What this solution (achieved 0.48811) has done: 'Your current score (0.48811) is substantially higher than the target (0.28755), so to move toward the target we should slightly *decrease* performance with minimal, low-risk changes that keep the same QA-span logic and submission validity. The smallest lever is to make the neutral shortcut less “correct”: instead of returning the full original tweet for neutral, return the cleaned text used for tokenization (whitespace-normalized), which typically reduces word-level Jaccard a bit without breaking formatting. Everything else (model, tokenization, start/end argmax selection, offset-based span extraction, submission schema) is kept identical to preserve core logic and ensure the pipeline remains stable and end-to-end.'
- What this solution (achieved 0.14468) has done: 'Your current score (0.48811) is well above the target (0.28755), so we should make a very small, low-risk change that *reduces* Jaccard while keeping the same model, same QA start/end argmax logic, and a valid submission. The simplest lever is the neutral-sentiment shortcut: predicting the full (cleaned) tweet is usually very strong, so we instead return a short neutral-like placeholder extracted from the same tweet (the first word), which typically lowers Jaccard but remains a valid span-like string. Everything else (tokenization, offset-based extraction for non-neutral, masking/softmax/argmax, and CSV writing) is left intact to preserve core semantics and stability. This should move the score downward toward the target band without risking runtime issues.'
- What this solution (achieved 0.48811) has done: 'Your current score (0.14468) is below the target (0.28755), so we should make the smallest change that plausibly increases Jaccard without changing the QA-span model/inference core. The biggest intentional score-degrader in your code is the neutral shortcut that returns only the first word; reverting that to the standard “return full original tweet text for neutral” typically boosts the metric a lot while keeping everything else identical. I only change that neutral handling and keep the same model, same softmax+argmax span selection, same offset-based extraction, and the same submission writing. This should move the score upward toward the target band with minimal risk.'
- What this solution (achieved 0.48811) has done: 'Your current score (0.48811) is well above the target (0.28755), so we should make a very small, controlled change that slightly degrades Jaccard without breaking the QA-span core logic or submission validity. The safest lever is the neutral shortcut: it’s usually very strong to output the full tweet, so we instead output the same whitespace-normalized text used for tokenization, which tends to reduce word-level Jaccard a bit while remaining a valid quoted string. Everything else (model, tokenization, softmax+argmax span selection, offset-based extraction, and CSV schema) is kept identical to preserve the core approach and runtime stability. This should move your score downward toward the target band with minimal risk.'
- What this solution (achieved 0.23734) has done: 'Your current score (0.48811) is well above the target (0.28755), so the goal is to *slightly reduce* performance with the smallest, safest change while keeping the exact same QA span model/inference pipeline and a valid submission. The most controlled lever is the neutral-sentiment shortcut, which is typically a large contributor to score; we make neutral predictions a bit less accurate by returning a shorter span-like string (first 3 whitespace tokens) from the same cleaned text, without touching tokenization, model, logits→softmax, argmax span selection, or offset-based extraction for non-neutral. This should move the Jaccard score downward toward the target band while preserving end-to-end stability and submission validity. Everything else remains identical.'
- What this solution (achieved 0.48811) has done: 'Your current score (0.23734) is below the target (0.28755), so we should make a small, controlled improvement without touching the model/QA-span mechanics. The biggest intentional degradation is the neutral-sentiment shortcut that returns only the first 3 words; reverting neutral handling to the standard “return the full original tweet text” typically increases word-level Jaccard and should move you closer to the target. Everything else (tokenization, softmax+argmax span selection, offset-based extraction, dataloader, and CSV writing) is kept identical to preserve core logic and runtime stability. This single change is the smallest lever likely to reduce the score gap.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import time
import math
import re
import string

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    DataCollatorWithPadding,
)

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
    MODEL_NAME = "microsoft/deberta-v3-base"
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
TEST_PATHS = [
    "/kaggle/input/tweet-sentiment-extraction/test.csv",
    "/kaggle/data/tweet-sentiment-extraction/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
]
test_path = next((p for p in TEST_PATHS if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError(
        f"Could not find test.csv in expected locations: {TEST_PATHS}"
    )

test_df = pd.read_csv(test_path)
test_df.head()



## === cell 4
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, use_fast=True)
CFG.TOKENIZER = tokenizer

collate_fn = DataCollatorWithPadding(
    CFG.TOKENIZER, padding="longest", return_tensors="pt"
)


class QADataset:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        orig_text = str(self.df.text.iloc[item])
        text = " ".join(orig_text.split())

        inputs = CFG.TOKENIZER(
            text,
            add_special_tokens=True,
            max_length=CFG.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        for k, v in inputs.items():
            if k in ("input_ids", "attention_mask", "token_type_ids"):
                inputs[k] = torch.tensor(v, dtype=torch.long)

        tok_text_tokens = CFG.TOKENIZER.convert_ids_to_tokens(
            inputs["input_ids"].tolist()
        )

        sentiment = [1, 0, 0]
        if self.df.sentiment.iloc[item] == "positive":
            sentiment = [0, 0, 1]
        if self.df.sentiment.iloc[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            "input_ids": inputs["input_ids"],
            "mask": inputs["attention_mask"],
            "offset_mapping": inputs["offset_mapping"],
            "text_tokens": " ".join(tok_text_tokens),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_text": self.df.text.iloc[item],
            "orig_sentiment": self.df.sentiment.iloc[item],
            "clean_text_for_tokenizer": text,
        }


def safe_collate(batch):
    """
    Fix DataLoader batching: stack tensors; keep offsets and strings as lists.
    """
    out = {}
    out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
    out["mask"] = torch.stack([b["mask"] for b in batch], dim=0)
    out["sentiment"] = torch.stack([b["sentiment"] for b in batch], dim=0)
    out["offset_mapping"] = [b["offset_mapping"] for b in batch]
    out["text_tokens"] = [b["text_tokens"] for b in batch]
    out["orig_text"] = [b["orig_text"] for b in batch]
    out["orig_sentiment"] = [b["orig_sentiment"] for b in batch]
    out["clean_text_for_tokenizer"] = [b["clean_text_for_tokenizer"] for b in batch]
    return out




## === cell 5
class QAModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                CFG.MODEL_NAME, output_hidden_states=True
            )
        else:
            if os.path.exists(config_path):
                self.config = torch.load(config_path, map_location="cpu")
            else:
                self.config = AutoConfig.from_pretrained(
                    CFG.MODEL_NAME, output_hidden_states=True
                )

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                CFG.MODEL_NAME, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        self.fc_dropout = nn.ModuleList([nn.Dropout(val) for val in CFG.FC_DROPOUT])
        self.fc = nn.Linear(self.config.hidden_size, 2)

    def forward(self, input_ids, mask):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=mask
        ).last_hidden_state
        logits = self.fc(embeddings)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 6
def test_fn(dataloader, model):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_mask = []
    fin_text_tokens = []
    fin_orig_text = []
    fin_orig_selected = []  # kept for signature compatibility (unused for test)
    fin_orig_sentiment = []
    fin_offsets = []
    fin_clean_text = []

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["mask"].to(device)
            text_tokens = data["text_tokens"]
            orig_text = data["orig_text"]
            orig_sentiment = data["orig_sentiment"]
            offsets = data["offset_mapping"]
            clean_texts = data["clean_text_for_tokenizer"]

            start_logits, end_logits = model(input_ids, mask)

            start_probs = torch.softmax(start_logits, dim=1)
            end_probs = torch.softmax(end_logits, dim=1)

            fin_output_start.append(start_probs.detach().cpu().numpy())
            fin_output_end.append(end_probs.detach().cpu().numpy())
            fin_mask.append(mask.detach().cpu().numpy())

            fin_text_tokens.extend(text_tokens)
            fin_orig_text.extend(orig_text)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_offsets.extend(offsets)
            fin_clean_text.extend(clean_texts)

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)
    fin_mask = np.vstack(fin_mask)

    return (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_text_tokens,
        fin_orig_text,
        fin_orig_selected,
        fin_orig_sentiment,
        fin_offsets,
        fin_clean_text,
    )




## === cell 7
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    collate_fn=safe_collate,
)



## === cell 8
possible_ckpt_roots = [
    "/kaggle/input/k/sagarikajadon/tweet-sentiment-extraction",
    "/kaggle/data/k/sagarikajadon/tweet-sentiment-extraction",
]
ckpt_root = next((p for p in possible_ckpt_roots if os.path.isdir(p)), None)

ckpt_paths = []
if ckpt_root is not None:
    for i in CFG.TRAIN_FOLDS:
        p = os.path.join(ckpt_root, f"QAbert{i}.pth")
        if os.path.exists(p):
            ckpt_paths.append((i, p))

config_path = "/kaggle/input/feedback-deberta-baseline-train/config.pth"

if len(ckpt_paths) > 0:
    for idx, (i, pth) in enumerate(ckpt_paths):
        model = QAModel(config_path=config_path, pretrained=False).to(device)
        state = torch.load(pth, map_location="cpu")
        model.load_state_dict(state, strict=True)

        if idx == 0:
            (
                fin_output_start,
                fin_output_end,
                fin_mask,
                fin_text_tokens,
                fin_orig_text,
                fin_orig_selected,
                fin_orig_sentiment,
                fin_offsets,
                fin_clean_text,
            ) = test_fn(test_loader, model)
        else:
            (
                a,
                b,
                c,
                fin_text_tokens,
                fin_orig_text,
                fin_orig_selected,
                fin_orig_sentiment,
                fin_offsets,
                fin_clean_text,
            ) = test_fn(test_loader, model)
            fin_output_start = np.add(fin_output_start, a)
            fin_output_end = np.add(fin_output_end, b)
            fin_mask = np.add(fin_mask, c)

    denom = float(len(ckpt_paths))
    fin_output_start = np.divide(fin_output_start, denom)
    fin_output_end = np.divide(fin_output_end, denom)
    fin_mask = np.divide(fin_mask, denom)
else:
    model = QAModel(config_path=None, pretrained=True).to(device)
    (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_text_tokens,
        fin_orig_text,
        fin_orig_selected,
        fin_orig_sentiment,
        fin_offsets,
        fin_clean_text,
    ) = test_fn(test_loader, model)




## === cell 9
def _span_from_offsets(clean_text, offsets, start_idx, end_idx):
    n = len(offsets)
    start_idx = int(max(0, min(start_idx, n - 1)))
    end_idx = int(max(0, min(end_idx, n - 1)))
    if end_idx < start_idx:
        end_idx = start_idx

    while start_idx < n and (offsets[start_idx] is None or offsets[start_idx][1] == 0):
        start_idx += 1
    while end_idx >= 0 and (offsets[end_idx] is None or offsets[end_idx][1] == 0):
        end_idx -= 1
    if end_idx < start_idx or start_idx >= n or end_idx < 0:
        return ""

    char_start = offsets[start_idx][0]
    char_end = offsets[end_idx][1]
    char_start = int(max(0, min(char_start, len(clean_text))))
    char_end = int(max(0, min(char_end, len(clean_text))))
    if char_end < char_start:
        char_end = char_start
    return clean_text[char_start:char_end].strip()


final_outputs = []

for j in range(len(fin_orig_text)):
    if str(fin_orig_sentiment[j]) == "neutral":
        final_outputs.append(str(fin_orig_text[j]).strip())
        continue

    mask_arr = fin_mask[j].astype(np.float32)

    start_probs = fin_output_start[j] * mask_arr
    end_probs = fin_output_end[j] * mask_arr

    idx_start = int(np.argmax(start_probs))
    idx_end = int(np.argmax(end_probs))
    if idx_end < idx_start:
        idx_end = idx_start

    pred = _span_from_offsets(
        clean_text=str(fin_clean_text[j]),
        offsets=fin_offsets[j],
        start_idx=idx_start,
        end_idx=idx_end,
    )

    if pred == "":
        pred = str(fin_orig_text[j]).strip()

    final_outputs.append(pred)

if len(final_outputs) != len(test_df):
    final_outputs = final_outputs[: len(test_df)] + [""] * max(
        0, len(test_df) - len(final_outputs)
    )



## === cell 10
test_ids = test_df["textID"].astype(str)
sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})
sub.head()



## === cell 11
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.isna().sum())
print(sub.head(3).to_string(index=False))
