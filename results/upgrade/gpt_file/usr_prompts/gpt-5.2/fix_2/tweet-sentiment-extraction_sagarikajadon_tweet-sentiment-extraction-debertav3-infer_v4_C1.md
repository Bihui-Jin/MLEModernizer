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

0.2895862758159637

# 6. Current score

0.48888

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.48888) has done: 'I fix the import-time crash caused by an incompatible `protobuf` version by avoiding importing `transformers` at module import time and instead using a local, lightweight tokenizer/model path check first. Then I make tokenizer/model loading robust in Kaggle by falling back to the public `microsoft/deberta-v3-base` (offline-friendly with `local_files_only=True`, otherwise normal) when the hardcoded dataset paths don’t exist. Finally, I ensure the dataloader returns a proper batched `text_tokens` list and make the post-processing safe (handle empty spans, keep lengths aligned) so a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import math
import re
import string
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
class CFG:
    DEBUG = False
    TRAIN = False  # inference-only notebook
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


seed_everything(CFG.SEED)



## === cell 3
TEST_PATH = "/kaggle/input/tweet-sentiment-extraction/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_df.head()



## === cell 4
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)
from transformers import DataCollatorWithPadding



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
TOKENIZER_LOCAL_PATH = (
    "/kaggle/input/k/sagarikajadon/tweet-sentiment-extraction/tokenizer/"
)


def load_tokenizer():
    if os.path.isdir(TOKENIZER_LOCAL_PATH):
        try:
            return AutoTokenizer.from_pretrained(TOKENIZER_LOCAL_PATH, use_fast=True)
        except Exception:
            pass
    try:
        return AutoTokenizer.from_pretrained(
            CFG.MODEL_NAME, use_fast=True, local_files_only=True
        )
    except Exception:
        return AutoTokenizer.from_pretrained(CFG.MODEL_NAME, use_fast=True)


tokenizer = load_tokenizer()
CFG.TOKENIZER = tokenizer

collate_fn = DataCollatorWithPadding(
    CFG.TOKENIZER, padding="longest", return_tensors="pt"
)




## === cell 6
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

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        tok_text_tokens = CFG.TOKENIZER.convert_ids_to_tokens(inputs["input_ids"])

        sentiment = [1, 0, 0]
        if self.df.sentiment.iloc[item] == "positive":
            sentiment = [0, 0, 1]
        if self.df.sentiment.iloc[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            "input_ids": input_ids,
            "mask": attention_mask,
            "text_tokens": " ".join(tok_text_tokens),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_text": self.df.text.iloc[item],
            "orig_sentiment": self.df.sentiment.iloc[item],
        }




## === cell 7
class QAModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                CFG.MODEL_NAME, output_hidden_states=True
            )
        else:
            if os.path.isfile(config_path):
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

        self.attention_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )

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




## === cell 8
def test_fn(dataloader, model):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_mask = []
    fin_text_tokens = []
    fin_orig_text = []
    fin_orig_selected = []
    fin_orig_sentiment = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            mask = data["mask"].to(device)

            text_tokens = data["text_tokens"]
            orig_text = data["orig_text"]
            orig_sentiment = data["orig_sentiment"]

            start_logits, end_logits = model(input_ids, mask)

            fin_output_start.append(start_logits.detach().cpu().numpy())
            fin_output_end.append(end_logits.detach().cpu().numpy())
            fin_mask.append(mask.detach().cpu().numpy())

            fin_text_tokens.extend(list(text_tokens))
            fin_orig_text.extend(list(orig_text))
            fin_orig_sentiment.extend(list(orig_sentiment))

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




## === cell 9
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), next(iter(test_loader))["input_ids"].shape



## === cell 10
CONFIG_PATH = "/kaggle/input/feedback-deberta-baseline-train/config.pth"
WEIGHTS_PATH = "/kaggle/input/k/sagarikajadon/tweet-sentiment-extraction/QAbert0.pth"

model = QAModel(config_path=CONFIG_PATH, pretrained=True).to(device)

if os.path.isfile(WEIGHTS_PATH):
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state, strict=False)

(
    fin_output_start,
    fin_output_end,
    fin_mask,
    fin_text_tokens,
    fin_orig_text,
    fin_orig_selected,
    fin_orig_sentiment,
) = test_fn(test_loader, model)

fin_output_start.shape, fin_output_end.shape, fin_mask.shape, len(fin_text_tokens)



## === cell 11
threshold = 0.3
final_outputs = []

for j in range(len(fin_text_tokens)):
    text_token = fin_text_tokens[j]
    mask = fin_mask[j]

    mask_start = fin_output_start[j] * mask
    mask_start = mask_start >= threshold

    mask_end = fin_output_end[j] * mask
    mask_end = mask_end >= threshold

    sel = [0] * len(mask)
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
        if 0 <= mj < len(sel):
            sel[mj] = 1

    toks = text_token.split()
    output_tokens = [x for i, x in enumerate(toks) if i < len(sel) and sel[i] == 1]
    output_tokens = [x for x in output_tokens if x not in ("[CLS]", "[SEP]")]

    if len(output_tokens) == 0:
        final_outputs.append(str(fin_orig_text[j]))
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

len(final_outputs), final_outputs[0]



## === cell 12
test_ids = test_df["textID"].astype(str).values
if len(final_outputs) != len(test_ids):
    if len(final_outputs) < len(test_ids):
        final_outputs = final_outputs + list(
            test_df["text"].iloc[len(final_outputs) :].astype(str).values
        )
    else:
        final_outputs = final_outputs[: len(test_ids)]

sub = pd.DataFrame(
    {"textID": test_ids, "selected_text": pd.Series(final_outputs, dtype="string")}
)
sub.head()



## === cell 13
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.columns.tolist())
print(sub.iloc[0].to_dict())
