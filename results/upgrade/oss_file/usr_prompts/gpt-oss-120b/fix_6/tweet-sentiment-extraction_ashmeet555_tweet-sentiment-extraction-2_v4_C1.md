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

3.14

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

0.587620198726654

# 6. Current score

0.48097

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.42661) has done: 'I fixed the tokenizer and model loading paths, replaced the custom weight loader with a standard HuggingFace DistilBERT Question‑Answering model, removed the faulty training loop, and added proper span extraction so the script now runs end‑to‑end and writes a valid `submission.csv` file.'
- What this solution (achieved 0.48144) has done: 'Implemented a minimal fix for the protobuf‑related import error by switching from the fast tokenizer to the standard `DistilBertTokenizer`. Adjusted the configuration to use a larger `MAX_LEN` (96) to capture more tweet content, which should modestly improve the Jaccard score while preserving the original model and training logic. No other logic changes were made, ensuring the pipeline runs end‑to‑end and produces a valid `submission.csv`.'
- What this solution (achieved 0.47296) has done: 'I added an environment‑variable guard before importing transformers to avoid the protobuf MessageFactory error, and modified the prediction loop to decode token IDs on the CPU (ensuring compatibility and correct span extraction). These small fixes let the notebook run end‑to‑end and keep the existing model logic while preserving the current score‑trajectory.'
- What this solution (achieved 0.42009) has done: 'The fix adds a monkey‑patch for the protobuf MessageFactory (providing a missing `GetPrototype` method) before importing any HuggingFace components, which removes the “MessageFactory has no attribute GetPrototype” error.  
It also raises `MAX_LEN` to 128 so the tokenizer keeps more of each tweet, a modest change that can lift the Jaccard score toward the target while preserving the original model‑based logic.'
- What this solution (achieved 0.48097) has done: 'The fix removes the problematic protobuf monkey‑patch that caused an import error, increases the tokenisation length to capture more tweet content, and improves span selection by ensuring the end index always follows the chosen start index (using a constrained argmax). These changes eliminate the runtime crash and give the model a better chance to predict longer, more accurate answer spans, moving the Jaccard score toward the target while keeping the original architecture unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
print("Libraries imported.")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class Config:
    MODEL_NAME = "distilbert-base-uncased"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 160
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-5




## === cell 2
print("Loading Data...")
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv").fillna("")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv").fillna("")

if Config.DEVICE.type == "cpu":
    print("⚠️ CPU DETECTED: Subsampling training data (not used for inference).")
    train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)

print(f"Training Data Shape: {train_df.shape}")
print(f"Test Data Shape: {test_df.shape}")




## === cell 3
class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row.text)
        sentiment = str(row.sentiment)
        encodings = self.tokenizer.encode_plus(
            sentiment,
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation="only_second",
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].flatten(),
            "attention_mask": encodings["attention_mask"].flatten(),
            "text": text,
            "sentiment": sentiment,
        }




## === cell 4
tokenizer = DistilBertTokenizer.from_pretrained(
    Config.MODEL_NAME, local_files_only=False
)




## === cell 5
train_dataset = TweetDataset(train_df, tokenizer, Config.MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

test_dataset = TweetDataset(test_df, tokenizer, Config.MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

print("DataLoaders created.")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")




## === cell 6
def load_pretrained_qa(model_name, device):
    print(f"Loading pretrained QA model '{model_name}'...")
    model = DistilBertForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model


model = load_pretrained_qa(Config.MODEL_NAME, Config.DEVICE)




## === cell 7
print("Starting Prediction Phase...")
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        sentiments = batch["sentiment"]
        original_texts = batch["text"]

        start_logits, end_logits = model(
            input_ids, attention_mask=attention_mask
        ).to_tuple()

        start_logits = start_logits.cpu().numpy()
        end_logits = end_logits.cpu().numpy()

        start_idxs = np.argmax(start_logits, axis=1)

        end_idxs = np.argmax(end_logits, axis=1)
        for i in range(len(end_idxs)):
            if end_idxs[i] < start_idxs[i]:
                end_candidate = (
                    np.argmax(end_logits[i, start_idxs[i] :]) + start_idxs[i]
                )
                end_idxs[i] = end_candidate

        input_ids_cpu = input_ids.cpu()

        for i in range(len(original_texts)):
            if sentiments[i] == "neutral":
                predictions.append(original_texts[i])
                continue

            span_ids = input_ids_cpu[i, start_idxs[i] : end_idxs[i] + 1]
            span_text = tokenizer.decode(
                span_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            predictions.append(span_text)

test_df["selected_text"] = predictions
submission = test_df[["textID", "selected_text"]]
submission.to_csv("submission.csv", index=False)
print("\n✅ SUCCESS: 'submission.csv' generated.")
