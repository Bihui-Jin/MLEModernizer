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

0.3447472751140594

# 6. Current score

0.17499

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.17499) has done: 'I fix the immediate runtime/import crash by removing the unused plotting/geo imports that trigger the protobuf `MessageFactory` error in this environment. Then I make the tokenizer/model loading robust: instead of referencing non-existent `/kaggle/input/k/...` files, it load DeBERTa tokenizer/model from `CFG.MODEL_NAME` with `local_files_only=True` and fall back to `local_files_only=False` if the environment allows it. Since no trained fold weights are available, I keep the same inference/core pipeline but run the model with its pretrained backbone so the notebook completes and produces a valid `submission.csv`. Finally, I fix small logic/shape issues (mask averaging, softmax dim, and always producing a non-empty selected_text) so the submission format is correct and stable.'
- What this solution (achieved 0.17499) has done: 'I fix the runtime crash caused by an incompatibility between `transformers` and the installed `protobuf` version (the `MessageFactory.GetPrototype` error) by forcing Transformers to use the pure-Python protobuf implementation before importing `transformers`. Then I correct the probability normalization: start/end logits should be softmaxed over the sequence length dimension (dim=1), not across the batch, which is a logic bug that hurts Jaccard score. Finally, I keep the rest of the pipeline identical and ensure the script still writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.17499) has done: 'I fix the protobuf/transformers crash by forcing the pure-Python protobuf implementation *and* clearing any preloaded `google.protobuf` modules before importing `transformers`, which is the direct cause of the `MessageFactory.GetPrototype` error. I also make the model/config/tokenizer loading robust in Kaggle by using the competition’s local dataset folder as a first-choice source (no internet needed) and only falling back to the hub if available. Additionally, I correct the incorrect checkpoint path so folds can actually load when present (this should materially improve score toward your target without changing the core model/inference logic). Finally, I keep the rest of the pipeline the same and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.17499) has done: 'I fix the `MessageFactory.GetPrototype` crash by ensuring the pure-Python protobuf runtime is selected *before* any protobuf/transformers modules are imported, and by restarting-cleaning any already-imported protobuf modules in a safer way. I also make tokenizer/model loading robust by preferring the local Kaggle competition dataset path (which doesn’t contain model files) only for data, while always loading the Hugging Face model by name (offline-first, hub fallback) to avoid mis-pointing `from_pretrained` at the dataset directory. These changes are runtime-stability fixes and keep the model/inference logic identical, so they should run end-to-end and produce `submission.csv`. With the crash removed, your existing softmax-dim fix remains in place, which should help move the score upward versus the broken run.'
- What this solution (achieved 0.17499) has done: 'I fix the protobuf/transformers `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *and* preventing any already-imported C++ protobuf modules from being reused before importing `transformers`. Then I make the DeBERTa config/model loading consistent by always using `local_files_only=True` first (to work offline) with a safe fallback to online only if available, which avoids intermittent import/load failures. Finally, I keep your existing inference logic intact (same model, same softmax-over-seq-length fix, same token-to-text reconstruction) and ensure the notebook completes and writes a valid `submission.csv`.'
- What this solution (achieved 0.17499) has done: 'I fix the persistent `MessageFactory.GetPrototype` crash by ensuring the pure-Python protobuf runtime is selected before any protobuf-dependent libraries are imported, and by fully clearing any already-loaded `google.protobuf` modules in a safer, more complete way. This change is purely a runtime stability fix and does not alter your model/inference logic. I also make the fold ensembling denominator reflect only successfully loaded checkpoints (otherwise missing weights unnecessarily dilute logits and can hurt score), while keeping the same ensembling approach. Finally, I keep the submission formatting identical but ensure the pipeline always runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.17499) has done: 'I fix the `MessageFactory.GetPrototype` crash by forcing Transformers to use the pure-Python protobuf runtime and preventing the incompatible C++ protobuf implementation from being used (this is the direct cause of your runtime error). I also make the import order safe by setting the relevant environment variables before any protobuf/transformers-related imports. These changes are runtime/stability-only and do not alter your model architecture, inference loop, or post-processing logic. With the crash removed, the script run end-to-end and write a valid `submission.csv` in the required format.'
- What this solution (achieved 0.17499) has done: 'I fix the persistent `MessageFactory.GetPrototype` crash by forcing Transformers to use the pure-Python protobuf runtime *before* any protobuf/transformers-related imports, and by setting additional environment variables that prevent the incompatible C++ protobuf backend from being used. Then I keep your same model/inference pipeline, but make the fold checkpoint loading more tolerant (`strict=False`) so mismatched keys won’t abort execution, which should also let any available weights load and improve score toward your target. Finally, I ensure the submission strings are always valid (non-empty) and the script writes `submission.csv` with the exact required columns.'
- What this solution (achieved 0.17499) has done: 'I fix the persistent `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf runtime before *any* protobuf/transformers import and by also setting `TRANSFORMERS_NO_PROTOBUF=1`, which avoids the incompatible compiled protobuf path in this Kaggle image. Then I make the fold checkpoint loader more robust to common checkpoint formats (raw `state_dict`, `model_state_dict`, `state_dict` with `module.` prefix) so any available weights can actually load and improve score toward your target without changing the model or inference logic. Finally, I keep your existing softmax-over-seq-length and reconstruction logic intact, only ensuring the pipeline runs end-to-end and writes a valid `submission.csv` with required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C"] = "1"
os.environ["TRANSFORMERS_NO_PROTOBUF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import importlib
import random
import string

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf") or m.startswith("protobuf"):
        del sys.modules[m]
importlib.invalidate_caches()

from transformers import AutoTokenizer, AutoModel, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
class CFG:
    DEBUG = False
    TRAIN = True
    N_FOLDS = 5
    TRAIN_FOLDS = [1]  # kept as-is; we will handle missing weights robustly
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
    "/kaggle/data/test.csv",
    "/kaggle/input/test.csv",
]

test_path = None
for p in TEST_PATHS:
    if os.path.exists(p):
        test_path = p
        break

if test_path is None:
    raise FileNotFoundError(
        f"Could not find test.csv in any expected location: {TEST_PATHS}"
    )

test_df = pd.read_csv(test_path)
test_df.head()




## === cell 4
tokenizer = None
tokenizer_errors = []

try:
    tokenizer = AutoTokenizer.from_pretrained(
        CFG.MODEL_NAME, use_fast=True, local_files_only=True
    )
except Exception as e:
    tokenizer_errors.append(("model_name_offline", repr(e)))

if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained(
        CFG.MODEL_NAME, use_fast=True, local_files_only=False
    )

CFG.TOKENIZER = tokenizer




## === cell 5
class QADataset:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = " ".join(str(self.df.text.iloc[item]).split())
        input_text = str(self.df.sentiment.iloc[item]) + "[SEP]" + text

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
        }




## === cell 6
class QAModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        if config_path is None:
            try:
                self.config = AutoConfig.from_pretrained(
                    CFG.MODEL_NAME, output_hidden_states=True, local_files_only=True
                )
            except Exception:
                self.config = AutoConfig.from_pretrained(
                    CFG.MODEL_NAME, output_hidden_states=True, local_files_only=False
                )
        else:
            self.config = torch.load(config_path)

        if pretrained:
            try:
                self.backbone = AutoModel.from_pretrained(
                    CFG.MODEL_NAME, config=self.config, local_files_only=True
                )
            except Exception:
                self.backbone = AutoModel.from_pretrained(
                    CFG.MODEL_NAME, config=self.config, local_files_only=False
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
            text_tokens = data["text_tokens"]
            orig_text = data["orig_text"]

            start_logits, end_logits = model(input_ids, mask)
            fin_output_start.append(start_logits.cpu())
            fin_output_end.append(end_logits.cpu())
            fin_mask.append(mask.cpu())

            fin_text_tokens.extend(list(text_tokens))
            fin_orig_text.extend(list(orig_text))

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




## === cell 9
def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(
        isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()
    ):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    return obj


def _safe_load_state_dict(model, path):
    if path is None or (not os.path.exists(path)):
        return False
    state = torch.load(path, map_location="cpu")
    state = _extract_state_dict(state)
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=False)
    return True


fin_output_start = None
fin_output_end = None
fin_mask = None
fin_text_tokens = None
fin_orig_text = None

loaded_any = False
num_loaded = 0

for idx, i in enumerate(CFG.TRAIN_FOLDS):
    model = QAModel(config_path=None, pretrained=True).to(device)

    ckpt_candidates = [
        f"/kaggle/input/tweet-sentiment-extraction/QAbert{i}.pth",
        f"/kaggle/data/tweet-sentiment-extraction/QAbert{i}.pth",
        f"/kaggle/input/tweet-sentiment-extraction/tweet-sentiment-extraction/QAbert{i}.pth",
        f"/kaggle/data/tweet-sentiment-extraction/tweet-sentiment-extraction/QAbert{i}.pth",
    ]
    ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)

    loaded = _safe_load_state_dict(model, ckpt_path)
    loaded_any = loaded_any or loaded
    if loaded:
        num_loaded += 1

    a, b, c, tt, ot = test_fn(test_loader, model)

    if idx == 0:
        fin_output_start, fin_output_end, fin_mask = a, b, c
        fin_text_tokens, fin_orig_text = tt, ot
    else:
        fin_output_start = fin_output_start + a
        fin_output_end = fin_output_end + b
        fin_mask = fin_mask + c  # will be averaged below

if fin_output_start is None:
    model = QAModel(config_path=None, pretrained=True).to(device)
    fin_output_start, fin_output_end, fin_mask, fin_text_tokens, fin_orig_text = (
        test_fn(test_loader, model)
    )

print("Loaded any fold checkpoint:", loaded_any)
print("Num loaded checkpoints:", num_loaded, "out of", len(CFG.TRAIN_FOLDS))




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
if loaded_any and num_loaded > 0:
    num_models = num_loaded
else:
    num_models = max(1, len(CFG.TRAIN_FOLDS))

s = torch.nn.Softmax(dim=1)
fin_output_start = s(fin_output_start / num_models)
fin_output_end = s(fin_output_end / num_models)
fin_mask = fin_mask / num_models




## === cell 11
final_outputs = []
for j in range(len(fin_text_tokens)):
    text_token = fin_text_tokens[j]
    mask = fin_mask[j].numpy().astype(np.float32)

    mask_start = fin_output_start[j].numpy() * mask
    mask_end = fin_output_end[j].numpy() * mask

    idx_start = int(np.argmax(mask_start))
    idx_end = int(np.argmax(mask_end))
    if idx_end < idx_start:
        idx_end = idx_start

    sel_mask = [0] * len(mask)
    for mj in range(idx_start, idx_end + 1):
        if 0 <= mj < len(sel_mask):
            sel_mask[mj] = 1

    output_tokens = [
        x
        for i, x in enumerate(text_token.split())
        if i < len(sel_mask) and sel_mask[i] == 1
    ]
    output_tokens = [
        x
        for x in output_tokens
        if x not in ("[CLS]", "[SEP]", "▁postive", "▁negative", "▁neutral")
    ]

    final_output = ""
    for ot in output_tokens:
        if ot.startswith("▁"):
            final_output = final_output + " " + ot[1:]
        elif len(ot) == 1 and ot in string.punctuation:
            final_output = final_output + ot
        else:
            final_output = final_output + " " + ot

    final_output = final_output.strip()
    if final_output == "":
        final_output = str(fin_orig_text[j]).strip()

    final_outputs.append(final_output)

len(final_outputs), final_outputs[0]




## === cell 12
test_ids = test_df["textID"].astype(str)
sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})
sub.head()




## === cell 13
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.isna().sum())
print(sub.head(3).to_string(index=False))
