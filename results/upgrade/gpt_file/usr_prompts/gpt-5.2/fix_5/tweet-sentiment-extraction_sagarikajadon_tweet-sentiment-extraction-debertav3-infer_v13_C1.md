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

0.25096

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.25096) has done: 'I fix the immediate import/runtime crash by pinning protobuf to the compatible pure-Python implementation and deferring `transformers` imports until after that environment fix. Then I remove the broken dependencies on missing Kaggle Dataset paths (`/kaggle/input/k/...`) by loading the tokenizer/model directly from `CFG.MODEL_NAME`, which is available in the Kaggle environment, so inference can run end-to-end. Finally, I correct a few inference-time logic/shape issues (softmax over the correct dimension, safe truncation, and consistent mask handling) so `final_outputs` is produced with the right length and a valid `submission.csv` is written.'
- What this solution (achieved 0.25096) has done: 'I fix the runtime crash coming from an incompatible protobuf implementation used by `transformers` by forcing the pure-Python protobuf backend *before* any transformers-related import and by guarding the import order. Then I correct two inference-time logic bugs that strongly hurt Jaccard: applying softmax over the wrong dimension (should be over sequence length, not batch) and using a proper tweet+sentiment separator (tokenizer’s `sep_token`) instead of the literal string `"[SEP]"`. These changes preserve the core model/inference approach (same backbone, same start/end argmax decoding), but should move the score substantially upward toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.25096) has done: 'I fix the `transformers` import/runtime crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend *and* disabling C++ protobuf before any `transformers` import, which is the typical cause in recent Python 3.11 Kaggle images. I also switch the model loading to use the actual fine-tuned QA head weights if they exist locally (common in Kaggle notebooks via a `pytorch_model.bin`/`model.safetensors`), otherwise fall back to the current base model so the notebook still runs end-to-end. Finally, I keep your decoding logic intact but ensure softmax is applied after masking and that the model is put in eval mode with deterministic settings preserved, producing a valid `submission.csv`.'
- What this solution (achieved 0.25096) has done: 'I fix the protobuf/transformers runtime crash by forcing the pure-Python protobuf backend *before* any `transformers` import and by avoiding the problematic `MessageFactory.GetPrototype` path via a safe fallback if needed. Then I ensure the model and tokenizer load correctly in the Kaggle environment and that inference runs end-to-end to produce `submission.csv`. Finally, I keep your start/end argmax decoding logic intact but make a minimal correction to use the tokenizer’s special tokens consistently when filtering output tokens (preventing accidental removal mismatches that severely hurt Jaccard). These changes are directly aimed at unblocking execution and improving the score from the random-head baseline toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import random
import time
import math
import re
import string

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
class CFG:
    DEBUG = False
    TRAIN = False  # inference-only
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
try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
except AttributeError as e:
    raise RuntimeError(
        "Transformers import failed due to protobuf runtime mismatch. "
        "Ensure PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is set before import."
    ) from e

test_path_candidates = [
    "/kaggle/input/tweet-sentiment-extraction/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/tweet-sentiment-extraction/test.csv",
    "/kaggle/data/test.csv",
]
test_path = None
for p in test_path_candidates:
    if os.path.exists(p):
        test_path = p
        break
if test_path is None:
    raise FileNotFoundError(
        f"Could not find test.csv in any of: {test_path_candidates}"
    )

test_df = pd.read_csv(test_path)
test_df.head()



## === cell 4
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, use_fast=True)
CFG.TOKENIZER = tokenizer




## === cell 5
class QADataset:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = " ".join(str(self.df.text.iloc[item]).split())
        sent = str(self.df.sentiment.iloc[item])

        sep = (
            CFG.TOKENIZER.sep_token if CFG.TOKENIZER.sep_token is not None else "[SEP]"
        )
        input_text = text + f" {sep} " + sent

        inputs = CFG.TOKENIZER(
            input_text,
            add_special_tokens=True,
            max_length=CFG.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        tok_text_tokens = CFG.TOKENIZER.convert_ids_to_tokens(inputs["input_ids"])

        return {
            "input_ids": input_ids,
            "mask": attention_mask,
            "text_tokens": " ".join(tok_text_tokens),
            "orig_text": self.df.text.iloc[item],
            "orig_sentiment": self.df.sentiment.iloc[item],
        }




## === cell 6
class QAModel(nn.Module):
    def __init__(self, config_path=None, pretrained=True):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                CFG.MODEL_NAME, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                CFG.MODEL_NAME, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        self.fc_dropout = nn.ModuleList([nn.Dropout(val) for val in CFG.FC_DROPOUT])
        self.fc = nn.Linear(self.config.hidden_size, 2)

    def forward(self, input_ids, mask, token_type_ids=None):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=mask
        ).last_hidden_state
        logits = None
        for dropout_layer in self.fc_dropout:
            out = self.fc(dropout_layer(embeddings))
            logits = out if logits is None else (logits + out)

        logits = logits / len(CFG.FC_DROPOUT)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




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

model = QAModel(config_path=None, pretrained=True).to(device)

ckpt_candidates = [
    "/kaggle/input/tweet-sentiment-extraction/pytorch_model.bin",
    "/kaggle/input/tweet-sentiment-extraction/model.bin",
    "/kaggle/input/tweet-sentiment-extraction/model.pth",
    "/kaggle/working/pytorch_model.bin",
    "/kaggle/working/model.bin",
    "/kaggle/working/model.pth",
]
ckpt_path = None
for p in ckpt_candidates:
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            for prefix in ("module.", "model."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
else:
    print(
        "No fine-tuned checkpoint found; using base pretrained backbone + random QA head."
    )

fin_output_start, fin_output_end, fin_mask, fin_text_tokens, fin_orig_text = test_fn(
    test_loader, model
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
fin_mask = fin_mask.float()

pad_mask = fin_mask == 0
fin_output_start = fin_output_start.masked_fill(pad_mask, -1e9)
fin_output_end = fin_output_end.masked_fill(pad_mask, -1e9)

s = torch.nn.Softmax(dim=-1)
fin_output_start = s(fin_output_start)
fin_output_end = s(fin_output_end)

(
    fin_output_start.shape,
    fin_output_end.shape,
    fin_mask.shape,
    len(fin_text_tokens),
    len(fin_orig_text),
)



## === cell 10
threshold = 0
final_outputs = []

cls_tok = CFG.TOKENIZER.cls_token if CFG.TOKENIZER.cls_token is not None else "[CLS]"
sep_tok = CFG.TOKENIZER.sep_token if CFG.TOKENIZER.sep_token is not None else "[SEP]"

for j in range(len(fin_text_tokens)):
    text_token = fin_text_tokens[j]
    mask_vec = fin_mask[j].numpy()

    start_prob = fin_output_start[j].numpy() * mask_vec
    end_prob = fin_output_end[j].numpy() * mask_vec

    idx_start = int(np.argmax(start_prob))
    idx_end = int(np.argmax(end_prob))
    if idx_end < idx_start:
        idx_end = idx_start

    tok_list = text_token.split()
    sel = [0] * len(tok_list)
    idx_end = min(idx_end, len(tok_list) - 1)
    idx_start = min(idx_start, len(tok_list) - 1)

    for mj in range(idx_start, idx_end + 1):
        if 0 <= mj < len(sel):
            sel[mj] = 1

    output_tokens = [x for i, x in enumerate(tok_list) if sel[i] == 1]
    output_tokens = [
        x
        for x in output_tokens
        if x not in (cls_tok, sep_tok, "▁postive", "▁negative", "▁neutral")
    ]

    final_output = ""
    for ot in output_tokens:
        if ot.startswith("▁"):
            final_output = final_output + " " + ot[1:]
        elif len(ot) == 1 and ot in string.punctuation:
            final_output = final_output + ot
        else:
            final_output = final_output + ot

    final_output = final_output.strip()
    if final_output == "":
        final_output = str(fin_orig_text[j]).strip()

    final_outputs.append(final_output)

len(final_outputs), final_outputs[:5]



## === cell 11
test_ids = test_df["textID"].astype(str).values
sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})

assert len(sub) == len(test_df)
sub.head()



## === cell 12
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.iloc[0])
