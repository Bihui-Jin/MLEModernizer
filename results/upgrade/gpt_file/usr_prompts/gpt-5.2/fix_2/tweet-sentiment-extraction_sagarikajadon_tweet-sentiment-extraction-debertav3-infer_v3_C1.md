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

0.21494

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.21494) has done: 'I fix the environment/import issue that prevents `transformers` from loading (protobuf / `MessageFactory.GetPrototype`), and make the tokenizer/model loading robust by falling back to the standard `CFG.MODEL_NAME` when the referenced Kaggle input artifacts don’t exist. I also fix missing imports and dataset class definition ordering so `QADataset` and `DataCollatorWithPadding` are available when used. Since your current run never produced a submission, I ensure inference always completes and writes a valid `submission.csv` with the required columns and row count. These changes keep the core DeBERTa QA-style start/end span logic intact while making it runnable end-to-end on the provided dataset paths.'

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    torch.backends.cudnn.deterministic = True


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
        text = " ".join(str(self.df.text.iloc[item]).split())

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
            "text_tokens": " ".join(tok_text_tokens),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_text": self.df.text.iloc[item],
            "orig_sentiment": self.df.sentiment.iloc[item],
        }




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

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["mask"].to(device)
            text_tokens = data["text_tokens"]
            orig_text = data["orig_text"]
            orig_sentiment = data["orig_sentiment"]

            start_logits, end_logits = model(input_ids, mask)

            fin_output_start.append(start_logits.detach().cpu().numpy())
            fin_output_end.append(end_logits.detach().cpu().numpy())
            fin_mask.append(mask.detach().cpu().numpy())

            fin_text_tokens.extend(text_tokens)
            fin_orig_text.extend(orig_text)
            fin_orig_sentiment.extend(orig_sentiment)

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
    )




## === cell 7
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
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
    ) = test_fn(test_loader, model)



## === cell 9
threshold = 0.3
final_outputs = []

for j in range(len(fin_text_tokens)):
    text_token = fin_text_tokens[j]
    mask_arr = fin_mask[j]

    mask_start = fin_output_start[j] * mask_arr
    mask_start = mask_start >= threshold

    mask_end = fin_output_end[j] * mask_arr
    mask_end = mask_end >= threshold

    mask = [0] * len(mask_arr)
    idx_start = np.nonzero(mask_start)[0]
    idx_end = np.nonzero(mask_end)[0]

    if len(idx_start) > 0:
        idx_start = int(idx_start[0])
        if len(idx_end) > 0:
            idx_end = int(idx_end[0])
        else:
            idx_end = idx_start
    else:
        idx_start = 0
        idx_end = 0

    if idx_end < idx_start:
        idx_end = idx_start

    for mj in range(idx_start, idx_end + 1):
        if mj < len(mask):
            mask[mj] = 1

    output_tokens = [
        x for i, x in enumerate(text_token.split()) if i < len(mask) and mask[i] == 1
    ]
    output_tokens = [x for x in output_tokens if x not in ("[CLS]", "[SEP]")]

    if len(output_tokens) == 0:
        final_outputs.append("")
        continue

    final_output = (
        output_tokens[0][1:] if output_tokens[0].startswith("▁") else output_tokens[0]
    )
    for ot in output_tokens[1:]:
        if ot.startswith("▁"):
            final_output = final_output + " " + ot[1:]
        elif len(ot) == 1 and ot in string.punctuation:
            final_output = final_output + ot
        else:
            final_output = final_output + ot

    final_outputs.append(final_output)

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
