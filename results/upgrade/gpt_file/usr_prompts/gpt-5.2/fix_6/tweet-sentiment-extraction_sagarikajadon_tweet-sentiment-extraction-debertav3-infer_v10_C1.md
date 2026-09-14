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

0.5212855339050293

# 6. Current score

0.16362

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18032) has done: 'I fix the environment import crash by avoiding the `transformers` optional `DataCollatorWithPadding` dependency that triggers the protobuf `MessageFactory` error in this Kaggle image, and by using a simple in-notebook padding-free collate. I also fix the incorrect tokenizer/model checkpoint paths by falling back to the competition’s standard model name (`microsoft/deberta-v3-base`) and running in pure inference mode without missing external `.pth` files. Since no valid submission was produced, the priority is to make the notebook run end-to-end and always write a correctly formatted `submission.csv` with `textID,selected_text`. The core span-extraction logic (start/end logits over token positions and decoding back to text) is preserved; the only model change is using an available pretrained backbone instead of nonexistent local weights so the pipeline can run and yield a reasonable baseline score.'
- What this solution (achieved 0.18032) has done: 'I fix the runtime crash caused by the `transformers` import path that pulls in an incompatible protobuf implementation (the `MessageFactory.GetPrototype` error) by forcing the pure-Python protobuf backend before importing `transformers`. I also make the DataLoader robust in Kaggle by setting `num_workers=0` to avoid multiprocessing/import side effects that can re-trigger the protobuf issue. These changes are execution-stability fixes and keep the same core inference-only DeBERTa span-extraction logic, ensuring a valid `submission.csv` is always written in the required format. No model/decoding logic is changed beyond what’s needed to run end-to-end.'
- What this solution (achieved 0.18032) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* any library that may import protobuf (including `transformers`) is imported, which requires moving those environment-variable lines to the very top of the script. I also harden execution by setting `TRANSFORMERS_NO_TF/NO_FLAX` and doing a small safe fallback to instantiate the backbone from config if `from_pretrained` still fails in this environment (keeps the same model/forward/decoding core logic, but prevents a hard crash). Finally, I ensure the submission is always written as `submission.csv` with the exact required columns and row alignment. These changes are primarily stability fixes and should allow the pretrained backbone to load correctly (which should move the score up substantially toward the target versus the current broken/unstable run).'
- What this solution (achieved 0.16362) has done: 'The crash comes from an incompatible protobuf backend being imported before/inside `transformers` when the model downloads/loads, so I force the pure-Python protobuf implementation *and* avoid the C++ extension explicitly via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before any transformer import. To keep the core span-extraction logic intact but move the score up toward the target, I minimally add the missing sentiment input by prepending the sentiment to the tweet text during tokenization (a standard requirement for this task) without changing the model head/training approach. I also harden model loading with a safe local-cache-first attempt and a deterministic fallback, so the notebook always completes and writes a valid `submission.csv`. Finally, I fix a small inefficiency/bug risk by applying softmax over the correct dimension for each example and ensuring masks align.'
- What this solution (achieved 0.16362) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by ensuring the pure-Python protobuf backend is forced before any indirect protobuf import and by importing `google.protobuf` immediately after setting env vars (this prevents the C++ backend from being loaded first). I also harden `transformers` loading by providing a stable fallback: if `AutoModel.from_pretrained` fails due to the protobuf issue, the code fall back to a randomly initialized `from_config` model so the notebook always completes and writes `submission.csv`. To move the score upward toward the target (without changing the core span-extraction approach), I load the correct span-extraction head weights from the Kaggle dataset if present (common in this competition as `.pth`/`.bin`), otherwise keep the existing behavior. These changes are minimal, keep the same forward/decoding logic, and primarily address the runtime blocker plus missing-weight issue that is holding the score far below target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

try:
    import google.protobuf  # noqa: F401
except Exception as _e:
    print("WARNING: protobuf import failed early:", repr(_e))

import gc
import random
import math
import re
import string
from pathlib import Path
import glob

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoConfig  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
class CFG:
    DEBUG = False
    TRAIN = False  # inference-only in this notebook
    N_FOLDS = 5
    TRAIN_FOLDS = [1]  # kept for compatibility; not used for checkpoint ensembling here
    SEED = 42
    TEST_BATCHSIZE = 64
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
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.SEED)




## === cell 3
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
test_df.head()




## === cell 4
try:
    tokenizer = AutoTokenizer.from_pretrained(
        CFG.MODEL_NAME, use_fast=True, local_files_only=True
    )
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, use_fast=True)

CFG.TOKENIZER = tokenizer




## === cell 5
class QADataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = " ".join(str(self.df.loc[item, "text"]).split())
        sent = self.df.loc[item, "sentiment"]

        text_with_sent = f"{sent} {text}"

        inputs = CFG.TOKENIZER(
            text_with_sent,
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
        if sent == "positive":
            sentiment = [0, 0, 1]
        elif sent == "negative":
            sentiment = [0, 1, 0]

        return {
            "input_ids": input_ids,
            "mask": attention_mask,
            "text_tokens": " ".join(tok_text_tokens),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_text": self.df.loc[item, "text"],
            "orig_sentiment": sent,
        }




## === cell 6
class QAModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                CFG.MODEL_NAME, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.backbone = None
            load_errors = []
            try:
                self.backbone = AutoModel.from_pretrained(
                    CFG.MODEL_NAME, config=self.config, local_files_only=True
                )
            except Exception as e:
                load_errors.append(("local_files_only", e))
            if self.backbone is None:
                try:
                    self.backbone = AutoModel.from_pretrained(
                        CFG.MODEL_NAME, config=self.config
                    )
                except Exception as e:
                    load_errors.append(("download", e))
            if self.backbone is None:
                print(
                    "WARNING: AutoModel.from_pretrained failed; falling back to from_config."
                )
                for tag, e in load_errors:
                    print(f" - from_pretrained ({tag}) exception:", repr(e))
                self.backbone = AutoModel.from_config(self.config)
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




## === cell 7
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
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["mask"].to(device)
            text_tokens = data["text_tokens"]
            orig_text = data["orig_text"]
            orig_sentiment = data["orig_sentiment"]

            start_logits, end_logits = model(input_ids, mask)

            fin_output_start.append(start_logits.detach().cpu())
            fin_output_end.append(end_logits.detach().cpu())
            fin_mask.append(mask.detach().cpu())

            fin_text_tokens.extend(list(text_tokens))
            fin_orig_text.extend(list(orig_text))
            fin_orig_sentiment.extend(list(orig_sentiment))

    fin_output_start = torch.vstack(fin_output_start)
    fin_output_end = torch.vstack(fin_output_end)
    fin_mask = torch.vstack(fin_mask)

    return (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_text_tokens,
        fin_orig_text,
        fin_orig_selected,
        fin_orig_sentiment,
    )




## === cell 8
test_dataset = QADataset(test_df)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)




## === cell 9
def _find_checkpoint():
    candidates = []
    search_roots = [
        "/kaggle/input/tweet-sentiment-extraction",
        "/kaggle/input",
        "/kaggle/working",
    ]
    patterns = ["*.pth", "*.pt", "*.bin"]
    for root in search_roots:
        for pat in patterns:
            candidates.extend(glob.glob(str(Path(root) / "**" / pat), recursive=True))
    prefer = []
    for p in candidates:
        name = Path(p).name.lower()
        if any(
            k in name
            for k in [
                "deberta",
                "roberta",
                "bert",
                "qa",
                "span",
                "start",
                "end",
                "model",
            ]
        ):
            prefer.append(p)
    return prefer[0] if prefer else (candidates[0] if candidates else None)


model = QAModel(config_path=None, pretrained=True).to(device)

ckpt_path = _find_checkpoint()
if ckpt_path is not None:
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if isinstance(state, dict):
            missing, unexpected = model.load_state_dict(state, strict=False)
            print("Loaded checkpoint:", ckpt_path)
            print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
        else:
            print(
                "Found checkpoint but unrecognized format (not a state_dict dict):",
                ckpt_path,
            )
    except Exception as e:
        print("WARNING: Failed to load checkpoint:", ckpt_path, "exception:", repr(e))
else:
    print(
        "No checkpoint found; using pretrained backbone + randomly initialized QA head."
    )

(
    fin_output_start,
    fin_output_end,
    fin_mask,
    fin_text_tokens,
    fin_orig_text,
    fin_orig_selected,
    fin_orig_sentiment,
) = test_fn(test_loader, model)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
s = torch.nn.Softmax(dim=1)
fin_output_start = s(fin_output_start)
fin_output_end = s(fin_output_end)
fin_mask = fin_mask.float()




## === cell 11
threshold = 0
final_outputs = []
for j in range(len(fin_text_tokens)):
    text_token = fin_text_tokens[j]
    mask = fin_mask[j].numpy()
    mask_start = (fin_output_start[j].numpy()) * mask
    mask_end = (fin_output_end[j].numpy()) * mask

    out_mask = [0] * len(mask)

    idx_start = int(np.argmax(mask_start))
    idx_end = int(np.argmax(mask_end))
    if idx_end < idx_start:
        idx_end = idx_start

    for mj in range(idx_start, idx_end + 1):
        out_mask[mj] = 1

    output_tokens = [
        x
        for i, x in enumerate(text_token.split())
        if i < len(out_mask) and out_mask[i] == 1
    ]
    output_tokens = [
        x for x in output_tokens if x not in ("[CLS]", "[SEP]", "<s>", "</s>")
    ]

    final_output = ""
    for ot in output_tokens:
        if ot.startswith("▁"):
            final_output = final_output + " " + ot[1:]
        elif len(ot) == 1 and ot in string.punctuation:
            final_output = final_output + ot
        else:
            final_output = final_output + " " + ot

    final_outputs.append(final_output.strip())




## === cell 12
if len(final_outputs) != len(test_df):
    final_outputs = list(test_df["text"].fillna("").astype(str).values)

test_ids = test_df["textID"]
sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})
sub.head()




## === cell 13
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.columns.tolist())
print(sub.iloc[0].to_dict())
