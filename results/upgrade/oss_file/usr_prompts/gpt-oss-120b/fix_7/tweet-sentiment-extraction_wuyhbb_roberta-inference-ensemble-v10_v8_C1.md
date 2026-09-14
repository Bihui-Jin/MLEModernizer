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

0.7285080552101135

# 6. Current score

0.60174

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61562) has done: 'Implemented a robust file‑search that includes Kaggle’s typical `/kaggle/input` and `/kaggle/working` directories, ensuring `test.csv` (and any other data files) are found without raising `FileNotFoundError`. This resolves the early import errors, defines `model_available`, and lets the subsequent cells run. The rest of the logic (heuristic selection and CSV export) remains unchanged, preserving the original approach while now producing a valid `submission.csv` that can be scored.'
- What this solution (achieved 0.60174) has done: 'I wrap the tokenizer and model loading in safe try/except blocks so that any protobuf‑related import errors are caught and the code falls back to the heuristic‑only path. Then I improve the heuristic: when a sentiment‑matching word is found, it now returns that word plus the following token (if present) to capture short phrases like “very good”, which should raise the Jaccard score toward the target while keeping the core logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, RobertaConfig


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


def find_file(relative_path: str) -> str:
    """
    Search common Kaggle directory layouts for a given file.
    """
    candidates = [
        relative_path,
        os.path.join("input", relative_path),
        os.path.join("data", relative_path),
        os.path.join("kaggle", "data", relative_path),
        os.path.join("kaggle", "data", "input", relative_path),
        os.path.join("data", "input", relative_path),
        os.path.join("working", relative_path),
        os.path.join("/kaggle", "input", relative_path),
        os.path.join("/kaggle", "working", relative_path),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Unable to locate {relative_path} in expected locations.")


test_path = find_file("test.csv")
test = pd.read_csv(test_path)

try:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
except Exception as e:
    print(f"Tokenizer loading failed ({e}); proceeding without tokenizer.")
    tokenizer = None

try:
    base_config = RobertaConfig.from_pretrained(
        "roberta-base", output_hidden_states=True
    )
    base_model = AutoModel.from_pretrained("roberta-base", config=base_config)
    model_available = True
except Exception as e:
    print(
        f"Model loading failed ({e}); proceeding with a fallback that uses the raw text."
    )
    base_model = None
    model_available = False




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class TweetModel(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.bert = pretrained_model
        hidden = self.bert.config.hidden_size
        self.cnn = nn.Conv1d(hidden * 3, hidden, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.whole_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden * 3, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )
        self.se_head = nn.Linear(hidden, 2)
        self.inst_head = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        _, pooled_output, hidden_states = self.bert(
            input_ids, attention_mask, token_type_ids=token_type_ids, return_dict=False
        )
        seq_output = torch.cat(
            [hidden_states[-1], hidden_states[-2], hidden_states[-3]], dim=-1
        )

        avg_output = torch.sum(seq_output * attention_mask.unsqueeze(-1), dim=1)
        avg_output = avg_output / torch.clamp(
            torch.sum(attention_mask, dim=1, keepdim=True), min=1e-9
        )
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0, 2, 1)).permute(0, 2, 1))
        se_out = self.se_head(self.dropout(seq_output))
        inst_out = self.inst_head(self.dropout(seq_output))
        return whole_out, se_out, se_out, inst_out


if model_available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TweetModel(base_model).to(device)
    model.eval()
else:
    model = None  # fallback placeholder
    device = torch.device("cpu")




## === cell 2
positive_words = {
    "good",
    "great",
    "nice",
    "love",
    "excellent",
    "best",
    "awesome",
    "happy",
    "amazing",
    "fantastic",
    "wonderful",
    "positive",
    "liked",
    "like",
}
negative_words = {
    "bad",
    "worst",
    "terrible",
    "hate",
    "awful",
    "sad",
    "angry",
    "poor",
    "disappointed",
    "negative",
    "hated",
    "hates",
    "unhappy",
}


def heuristic_selected(row):
    sentiment = str(row["sentiment"]).strip().lower()
    text = str(row["text"])
    if sentiment == "neutral":
        return text  # neutral case – return whole tweet
    tokens = text.split()
    target_set = positive_words if sentiment == "positive" else negative_words
    for idx, tok in enumerate(tokens):
        clean_tok = tok.lower().strip(".,!?\"'")
        if clean_tok in target_set:
            if idx + 1 < len(tokens):
                return f"{tok} {tokens[idx + 1]}"
            else:
                return tok
    return text


test["selected_text"] = test.apply(heuristic_selected, axis=1)




## === cell 3
submission_path = "submission.csv"
test[["textID", "selected_text"]].to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
