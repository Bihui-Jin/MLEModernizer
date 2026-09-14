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

0.7285080552101135

# 6. Current score

0.60584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61562) has done: 'Implemented a robust file‑search that includes Kaggle’s typical `/kaggle/input` and `/kaggle/working` directories, ensuring `test.csv` (and any other data files) are found without raising `FileNotFoundError`. This resolves the early import errors, defines `model_available`, and lets the subsequent cells run. The rest of the logic (heuristic selection and CSV export) remains unchanged, preserving the original approach while now producing a valid `submission.csv` that can be scored.'
- What this solution (achieved 0.60174) has done: 'I wrap the tokenizer and model loading in safe try/except blocks so that any protobuf‑related import errors are caught and the code falls back to the heuristic‑only path. Then I improve the heuristic: when a sentiment‑matching word is found, it now returns that word plus the following token (if present) to capture short phrases like “very good”, which should raise the Jaccard score toward the target while keeping the core logic unchanged.'
- What this solution (achieved 0.60214) has done: 'I wrapped the transformer imports in a safe try/except block so the notebook no longer crashes when protobuf incompatibilities arise, and I enhanced the heuristic to also capture the preceding token (when present) along with the matched sentiment word and its following token. This yields a slightly larger selected phrase (e.g., “very good”) which improves the Jaccard score while keeping the original fallback‑only approach.'
- What this solution (achieved 0.60543) has done: 'I keep the existing data loading and model‑fallback logic, but replace the heuristic with a broader span selection: locate all sentiment‑matched tokens, then return the text from one token before the first match to one token after the last match (clamped to the tweet bounds). This adds a few extra relevant words, which should raise the Jaccard score toward the target while preserving the original fallback‑only approach.'
- What this solution (achieved 0.61579) has done: 'I tighten the heuristic so it returns only the exact span that contains the sentiment‑matching tokens (removing the extra surrounding words that were lowering the Jaccard score) and keep the robust fallback logic that avoids transformer loading errors. This change preserves the overall pipeline while improving the alignment between predicted and true selected text, moving the score closer to the target.'
- What this solution (achieved 0.60359) has done: 'I enhance the heuristic to capture surrounding intensifiers and extend the selected span to the next punctuation token, which should better match the true selected text and raise the Jaccard score toward the target. The fix also adds a small safety check for empty tokens. No core model logic is altered.'
- What this solution (achieved 0.60604) has done: 'Implemented a tighter heuristic for selecting the sentiment‑matching span: it now picks the first matching word, optionally prefixes an intensifier, and adds the immediate following token (unless it’s punctuation). This reduces over‑selection and aligns predictions more closely with the true excerpts, which should improve the Jaccard score while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.60359) has done: 'Implemented an enhanced heuristic for selecting the sentiment‑matching span. The new logic expands the span to include surrounding intensifiers and continues adding tokens until punctuation is encountered, and also handles multiple sentiment words by spanning from the first to the last match. This modest change stays within the original fallback‑only approach while improving alignment with the true selected text, helping move the Jaccard score toward the target. No other core logic was altered, and the script now reliably writes a proper `submission.csv`.'
- What this solution (achieved 0.60584) has done: 'The fix removes the problematic transformer imports (which caused a protobuf AttributeError) and switches to a safe fallback, while the heuristic is rewritten to extract the selected span directly from the original text using regex. This preserves the original pipeline but yields a more accurate substring, improving the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

AutoTokenizer = None
AutoModel = None
RobertaConfig = None


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

model_available = False
base_model = None




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
        return whole_out, se_out, inst_out


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
intensifiers = {
    "very",
    "so",
    "really",
    "extremely",
    "quite",
    "absolutely",
    "amazingly",
    "incredibly",
    "totally",
    "utterly",
    "pretty",
    "somewhat",
}


def heuristic_selected(row):
    sentiment = str(row["sentiment"]).strip().lower()
    text = str(row["text"])
    if sentiment == "neutral":
        return text

    target_set = positive_words if sentiment == "positive" else negative_words

    word_iters = list(re.finditer(r"\b\w+\b", text))
    match_word_idxs = [
        i for i, m in enumerate(word_iters) if m.group(0).lower() in target_set
    ]

    if not match_word_idxs:
        return text

    first_idx = match_word_idxs[0]
    start_pos = word_iters[first_idx].start()
    if first_idx > 0:
        prev_word = word_iters[first_idx - 1].group(0).lower()
        if prev_word in intensifiers:
            start_pos = word_iters[first_idx - 1].start()

    last_idx = match_word_idxs[-1]
    end_pos = word_iters[last_idx].end()
    while end_pos < len(text) and text[end_pos] not in ".!?":
        end_pos += 1
    if end_pos < len(text) and text[end_pos] in ".!?":
        end_pos += 1

    return text[start_pos:end_pos].strip()


test["selected_text"] = test.apply(heuristic_selected, axis=1)



## === cell 3
submission_path = "submission.csv"
test[["textID", "selected_text"]].to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
