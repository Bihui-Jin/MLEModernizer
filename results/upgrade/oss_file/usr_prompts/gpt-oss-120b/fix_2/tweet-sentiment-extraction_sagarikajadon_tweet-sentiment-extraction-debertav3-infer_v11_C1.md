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

0.14593

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14593) has done: 'I fixed the protobuf import issue, corrected the tokenizer and model loading paths, added safe fallbacks when pretrained weights are missing, and replaced the final post‑processing with a simple but valid approach that outputs the original tweet text as the selected snippet. This ensures the script runs end‑to‑end and creates a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
import os, gc, random, time, math, re, string, warnings
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")




## === cell 1
class CFG:
    DEBUG = False
    TRAIN = True
    N_FOLDS = 5
    TRAIN_FOLDS = [1]  # keep a single fold for inference
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.SEED)


## === cell 3
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")


## === cell 4
from transformers import AutoTokenizer, AutoConfig, AutoModel

CFG.TOKENIZER = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, use_fast=True)




## === cell 5
class QADataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = " ".join(str(self.df.text.iloc[idx]).split())
        input_text = self.df.sentiment.iloc[idx] + "[SEP]" + text
        inputs = CFG.TOKENIZER(
            input_text,
            add_special_tokens=True,
            max_length=CFG.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        tok_text_tokens = CFG.TOKENIZER.convert_ids_to_tokens(inputs["input_ids"])
        return {
            "input_ids": inputs["input_ids"],
            "mask": inputs["attention_mask"],
            "text_tokens": " ".join(tok_text_tokens),
            "orig_text": self.df.text.iloc[idx],
        }




## === cell 6
class QAModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            CFG.MODEL_NAME, output_hidden_states=True
        )
        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                CFG.MODEL_NAME, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)
        self.fc_dropout = nn.ModuleList([nn.Dropout(p) for p in CFG.FC_DROPOUT])
        self.fc = nn.Linear(self.config.hidden_size, 2)

    def forward(self, input_ids, mask):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=mask
        ).last_hidden_state
        logits = self.fc(embeddings)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 7
def test_fn(dataloader, model):
    model.eval()
    all_start, all_end, all_mask = [], [], []
    all_text_tokens, all_orig_text = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            start_logits, end_logits = model(input_ids, mask)

            all_start.append(start_logits.cpu())
            all_end.append(end_logits.cpu())
            all_mask.append(mask.cpu())
            all_text_tokens.extend(batch["text_tokens"])
            all_orig_text.extend(batch["orig_text"])
    return (
        torch.vstack(all_start),
        torch.vstack(all_end),
        torch.vstack(all_mask),
        all_text_tokens,
        all_orig_text,
    )




## === cell 8
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


## === cell 9
model = QAModel(pretrained=True).to(device)
checkpoint_path = "/kaggle/input/k/sagarikajadon/tweet-sentiment-extraction/QAbert1.pth"
if os.path.exists(checkpoint_path):
    try:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print("Loaded pretrained checkpoint.")
    except Exception as e:
        print(f"Failed loading checkpoint ({e}), using raw pretrained model.")
else:
    print("Checkpoint not found, using raw pretrained model.")

fin_output_start, fin_output_end, fin_mask, fin_text_tokens, fin_orig_text = test_fn(
    test_loader, model
)

softmax = torch.nn.Softmax(dim=1)
start_probs = softmax(fin_output_start)
end_probs = softmax(fin_output_end)

final_outputs = []
for i in range(len(fin_text_tokens)):
    mask = fin_mask[i].bool()
    start_p = start_probs[i] * mask
    end_p = end_probs[i] * mask

    start_idx = torch.argmax(start_p).item()
    end_idx = torch.argmax(end_p).item()
    if end_idx < start_idx:
        end_idx = start_idx

    tokens = fin_text_tokens[i].split()
    selected = tokens[start_idx : end_idx + 1]
    selected = [t for t in selected if t not in ("[CLS]", "[SEP]")]
    out = ""
    for tok in selected:
        if tok.startswith("▁"):
            out += " " + tok[1:]
        elif len(tok) == 1 and tok in string.punctuation:
            out += tok
        else:
            out += " " + tok
    final_outputs.append(out.strip() if out else fin_orig_text[i])


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
submission = pd.DataFrame({"textID": test_df["textID"], "selected_text": final_outputs})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
