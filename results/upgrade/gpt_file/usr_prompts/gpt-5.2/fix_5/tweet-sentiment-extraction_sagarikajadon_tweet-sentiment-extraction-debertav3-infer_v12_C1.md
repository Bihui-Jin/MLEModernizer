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

0.717913031578064

# 6. Current score

0.4734

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.26939) has done: 'I fix the environment/runtime crash caused by an incompatible protobuf version by avoiding unnecessary imports (notably `pandas_profiling`) and using safe, minimal imports. Then I remove hard-coded Kaggle Dataset paths that don’t exist in your environment and instead load the tokenizer/model directly from `CFG.MODEL_NAME` with `local_files_only=True` fallback to online if available. Finally, I make the inference loop robust (handle missing checkpoints, ensure tensors exist, correct softmax dimension, and always generate `final_outputs` of the right length) and write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.01047) has done: 'I fix the runtime crash coming from a protobuf/transformers import incompatibility by avoiding the code path that triggers it and by switching to a locally-available backbone that doesn’t rely on the problematic dependency chain. Then I fix a scoring-critical bug in post-processing: applying softmax over the wrong dimension (it must be over the token dimension), which currently makes the span selection essentially random and explains the very low Jaccard score. Finally, I correct the checkpoint directory to a real path in this environment (or gracefully fall back), keep the core span-extraction logic intact, and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.4734) has done: 'I fix the crash by removing the `torch.hub` fairseq dependency (it requires `hydra-core`, which isn’t installed) and instead load the same `roberta-base` backbone via `transformers`, keeping the exact same “backbone → dropout ensemble → linear head → start/end logits” core logic. I also fix the DataLoader collation issue by returning `text_tokens` as a list of tokens (so batching works), and keep the softmax applied over the token dimension (sequence length). Finally, I ensure inference always runs end-to-end on CPU/GPU and writes a valid `submission.csv` with `textID,selected_text`.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import string

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    torch.backends.cudnn.deterministic = True


seed_everything(seed=CFG.SEED)



## === cell 3
TEST_PATH = "/kaggle/input/tweet-sentiment-extraction/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_df.head()



## === cell 4

from transformers import AutoTokenizer, AutoModel, AutoConfig


def load_transformers_roberta(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
    backbone = AutoModel.from_pretrained(model_name, config=config)
    backbone.eval()
    return tokenizer, backbone


CFG.TOKENIZER, backbone_model = load_transformers_roberta(CFG.MODEL_NAME)

CFG.PAD_ID = int(CFG.TOKENIZER.pad_token_id)
CFG.BOS_ID = (
    int(CFG.TOKENIZER.bos_token_id)
    if CFG.TOKENIZER.bos_token_id is not None
    else int(CFG.TOKENIZER.cls_token_id)
)
CFG.EOS_ID = (
    int(CFG.TOKENIZER.eos_token_id)
    if CFG.TOKENIZER.eos_token_id is not None
    else int(CFG.TOKENIZER.sep_token_id)
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
class QADataset:
    def __init__(self, df):
        self.df = df

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
            return_tensors="pt",
        )

        ids = enc["input_ids"].squeeze(0).long()
        attention_mask = enc["attention_mask"].squeeze(0).long()

        tok_text_tokens = CFG.TOKENIZER.convert_ids_to_tokens(ids.tolist())

        return {
            "input_ids": ids,
            "mask": attention_mask,
            "text_tokens": tok_text_tokens,
            "orig_text": self.df.text.iloc[item],
            "orig_sentiment": self.df.sentiment.iloc[item],
        }




## === cell 6
class QAModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = backbone_model
        self.hidden_size = int(self.backbone.config.hidden_size)

        self.fc_dropout = nn.ModuleList([nn.Dropout(val) for val in CFG.FC_DROPOUT])
        self.fc = nn.Linear(self.hidden_size, 2)

        self.pretrained = pretrained

    def forward(self, input_ids, mask, token_type_ids=None):
        out = self.backbone(input_ids=input_ids, attention_mask=mask)
        embeddings = out.last_hidden_state  # [B, T, H]

        logits = None
        for dropout_layer in self.fc_dropout:
            cur = self.fc(dropout_layer(embeddings))
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
def test_fn(dataloader, model):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_mask = []
    fin_text_tokens = []
    fin_orig_text = []

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["mask"].to(device)

            start_logits, end_logits = model(input_ids, mask)

            fin_output_start.append(start_logits.detach().cpu())
            fin_output_end.append(end_logits.detach().cpu())
            fin_mask.append(mask.detach().cpu())

            fin_text_tokens.extend(data["text_tokens"])
            fin_orig_text.extend(data["orig_text"])

    fin_output_start = torch.vstack(fin_output_start)
    fin_output_end = torch.vstack(fin_output_end)
    fin_mask = torch.vstack(fin_mask)

    return fin_output_start, fin_output_end, fin_mask, fin_text_tokens, fin_orig_text




## === cell 8
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model = QAModel(pretrained=True).to(device)
fin_output_start, fin_output_end, fin_mask, fin_text_tokens, fin_orig_text = test_fn(
    test_loader, model
)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 9
fin_output_start = torch.softmax(fin_output_start, dim=1)
fin_output_end = torch.softmax(fin_output_end, dim=1)



## === cell 10
final_outputs = []
for j in range(len(fin_text_tokens)):
    toks = fin_text_tokens[j]
    mask = fin_mask[j].numpy().astype(np.float32)

    start_probs = fin_output_start[j].numpy() * mask
    end_probs = fin_output_end[j].numpy() * mask

    idx_start = int(np.argmax(start_probs))
    idx_end = int(np.argmax(end_probs))
    if idx_end < idx_start:
        idx_end = idx_start

    token_keep = np.zeros_like(mask, dtype=np.int32)
    token_keep[idx_start : idx_end + 1] = 1

    output_tokens = [
        x for i, x in enumerate(toks) if i < len(token_keep) and token_keep[i] == 1
    ]

    special_set = {
        CFG.TOKENIZER.cls_token,
        CFG.TOKENIZER.sep_token,
        CFG.TOKENIZER.pad_token,
        "<s>",
        "</s>",
        "<pad>",
        "[CLS]",
        "[SEP]",
        "▁postive",
        "▁positive",
        "▁negative",
        "▁neutral",
        "positive",
        "negative",
        "neutral",
    }
    output_tokens = [x for x in output_tokens if x not in special_set and x is not None]

    final_output = CFG.TOKENIZER.convert_tokens_to_string(output_tokens).strip()

    if final_output == "":
        final_output = str(fin_orig_text[j]).strip()

    final_outputs.append(final_output)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/2308583540.py in <cell line: 0>()
      3 for j in range(len(fin_text_tokens)):
      4     toks = fin_text_tokens[j]
----> 5     mask = fin_mask[j].numpy().astype(np.float32)
      6 
      7     start_probs = fin_output_start[j].numpy() * mask

IndexError: index 2749 is out of bounds for dimension 0 with size 2749

## === cell 11
assert len(final_outputs) == len(test_df), (len(final_outputs), len(test_df))

sub = pd.DataFrame({"textID": test_df["textID"].values, "selected_text": final_outputs})
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
