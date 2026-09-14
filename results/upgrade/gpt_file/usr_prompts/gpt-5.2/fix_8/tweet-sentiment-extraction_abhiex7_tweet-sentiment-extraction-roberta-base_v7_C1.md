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

No external packages required in the script and installed.

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

0.6881447434425354

# 6. Current score

0.46871

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.47894) has done: 'I remove the directory-walk print that triggers an environment protobuf/transformers AttributeError, and I fix the tokenizer initialization to use the correct `tokenizers` API so `TOKENIZER` is defined. I also make the script robust to missing external RoBERTa weight folders by using the Kaggle-provided `roberta-base` from `transformers` when those paths don’t exist, while keeping the same model forward/head logic and inference flow. Then I fix the broken triple-quoted training cell so the notebook runs end-to-end (training remains optional), ensure `SEED`/device are defined before use, and finally guarantee a valid `submission.csv` is written with the required columns and row alignment.'
- What this solution (achieved 0.47894) has done: 'I fix the runtime import crash by avoiding the `tokenizers`/protobuf path that triggers `MessageFactory.GetPrototype` in this environment, while keeping the same RoBERTa-based model and training/inference flow. Concretely, I switch the dataset encoding to use `RobertaTokenizerFast` offsets (equivalent semantics for spans) and keep the same input construction and offset-based span extraction. This should both unblock execution and substantially improve score versus the current “untrained/random weights” behavior by correctly loading `roberta-base` tokenizer/model without hitting the protobuf issue. Finally, I keep the submission formatting and alignment checks, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.47894) has done: 'I fix the immediate import crash (`MessageFactory.GetPrototype`) by pinning protobuf to the pure-Python implementation before `transformers` is imported, which avoids the known C++ protobuf incompatibility in some Kaggle images. Then I keep your RoBERTa/tokenizer + offset-span extraction logic intact, but ensure the model is loaded with pretrained `roberta-base` weights (instead of random-init) by constructing it via `RobertaModel.from_pretrained` inside `TweetModel`. Finally, I make inference deterministic and robust (CPU-safe offsets conversion, proper eval/no-grad) and keep the submission formatting exactly as required so `submission.csv` is always produced and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.47894) has done: 'We fix the immediate crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend *before* anything that might import protobuf/transformers, and by importing `transformers` after that environment variable is set. Then we keep your exact RoBERTa + offset-span extraction model logic, but ensure inference uses pretrained `roberta-base` weights (not random init) and runs deterministically with `eval()`/`no_grad()`, which should raise the score substantially toward the target. Finally, we keep the submission formatting unchanged but make sure row alignment/length always matches `sample_submission.csv` and `submission.csv` is always written.'
- What this solution (achieved 0.46871) has done: 'I fix the immediate crash caused by an incompatible protobuf backend by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before importing `transformers`. Then I keep your exact RoBERTa + offset-span extraction pipeline, but ensure inference is stable by clamping predicted start/end indices to the real (unpadded) token span derived from offsets—this is a minimal post-processing fix that typically improves Jaccard without changing the model itself. Finally, I keep the same submission-writing logic while ensuring the output is always a valid `submission.csv` with correct row count and columns.'
- What this solution (achieved 0.46871) has done: 'I fix the immediate `protobuf`/`MessageFactory.GetPrototype` crash by ensuring the pure-Python protobuf backend is selected *before* any indirect protobuf import and by avoiding an unconditional `transformers` import at module load time. To keep your core RoBERTa + offset-span extraction logic unchanged while improving score toward the target, I also load the official pretrained `roberta-base` model weights reliably (offline-safe) and keep deterministic inference with proper `eval()`/`no_grad()`. Finally, I make the dataloading/inference more robust in Kaggle (workers/pinning) and ensure `submission.csv` is always produced with correct row alignment and required columns.'
- What this solution (achieved 0.46871) has done: 'I fix the runtime crash coming from an incompatible protobuf backend by forcing the pure-Python protobuf implementation before importing `transformers`, and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early and consistently. Then I keep your RoBERTa + offset-span extraction logic intact, but ensure inference uses deterministic/eval-safe settings and doesn’t inadvertently trigger the problematic protobuf path during import. Finally, I keep the same submission formatting, while guaranteeing the script always writes a valid `submission.csv` with the required columns and row alignment.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold

from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast


def seedall(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_PATH = "../input/tweet-sentiment-extraction/train.csv"
TEST_PATH = "../input/tweet-sentiment-extraction/test.csv"
SAMPLE_SUB_PATH = "../input/tweet-sentiment-extraction/sample_submission.csv"

MAX_LEN = 192
PATH = "/kaggle/input/roberta/"
ROBERTAFOLD = "/kaggle/input/robertafolds4/"

EPOCHS = 3
BATCH_SIZE = 32
SEED = 42
DROPOUT = 0.2
LEARNING_RATE = 4e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seedall(SEED)

train_df = pd.read_csv(TRAIN_PATH)
train_df["text"] = train_df["text"].astype(str)
train_df["selected_text"] = train_df["selected_text"].astype(str)



## === cell 2
_hf_tok = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
RobertaConf = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=True)




## === cell 3
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.max_len = MAX_LEN
        self.labeled = "selected_text" in df.columns
        self.hf_tokenizer = _hf_tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet, offsets = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweet"] = tweet
        data["offsets"] = offsets

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx

        return data

    def _encode(self, text: str):
        out = self.hf_tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return out["input_ids"], out["offset_mapping"]

    def get_input_data(self, row):
        tweet = " " + " ".join(row.text.lower().split())

        encoding_ids, encoding_offsets = self._encode(tweet)
        sentiment_ids, _ = self._encode(str(row.sentiment))

        ids = [0] + sentiment_ids + [2, 2] + list(encoding_ids) + [2]
        offsets = [(0, 0)] * 4 + list(encoding_offsets) + [(0, 0)]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len
        else:
            ids = ids[: self.max_len]
            offsets = offsets[: self.max_len]

        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != 1).long()
        offsets = torch.tensor(offsets, dtype=torch.long)
        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(str(row.selected_text).lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        if len(selected_text) > 1:
            for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
                if " " + tweet[ind : ind + len_st] == selected_text:
                    idx0 = ind
                    idx1 = ind + len_st - 1
                    break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        offsets_np = offsets.numpy()
        for j, (offset1, offset2) in enumerate(offsets_np):
            if offset2 > offset1 and sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        if len(target_idx) == 0:
            non_pad = np.where(offsets_np[:, 1] > 0)[0]
            if len(non_pad) == 0:
                return torch.tensor(0), torch.tensor(0)
            start_idx = int(non_pad.min())
            end_idx = int(non_pad.max())
            return torch.tensor(start_idx), torch.tensor(end_idx)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return torch.tensor(start_idx), torch.tensor(end_idx)


def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    train_df_ = df.iloc[train_idx].reset_index(drop=True)
    val_df_ = df.iloc[val_idx].reset_index(drop=True)

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df_),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df_),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return {"train": train_loader, "val": val_loader}


def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df.reset_index(drop=True)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader




## === cell 4
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        config = RobertaConf

        self.roberta = RobertaModel.from_pretrained("roberta-base", config=config)

        m_size = 128
        self.qa_outputs1c = torch.nn.Conv1d(config.hidden_size, m_size, kernel_size=1)
        self.qa_outputs2c = torch.nn.Conv1d(config.hidden_size, m_size, kernel_size=1)

        self.qa_outputs1 = nn.Linear(m_size, 1)
        self.qa_outputs2 = nn.Linear(m_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.leaky1 = nn.LeakyReLU(0.2, True)
        self.leaky2 = nn.LeakyReLU(0.2, True)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids, attention_mask=attention_mask)

        s_out = self.dropout(out[0])
        s_out = torch.nn.functional.pad(s_out.transpose(1, 2), (1, 0))

        out1 = self.qa_outputs1c(s_out).transpose(1, 2)
        out2 = self.qa_outputs2c(s_out).transpose(1, 2)
        out1 = self.leaky1(out1)
        out2 = self.leaky1(out2)
        start_logits = self.qa_outputs1(self.dropout(out1)).squeeze(-1)
        end_logits = self.qa_outputs2(self.dropout(out2)).squeeze(-1)
        return start_logits, end_logits




## === cell 5
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    return start_loss + end_loss




## === cell 6
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)
    true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, pred)




## === cell 7
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    model.to(DEVICE)

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0

            for data in dataloaders_dict[phase]:
                ids = data["ids"].to(DEVICE)
                masks = data["masks"].to(DEVICE)
                tweet = data["tweet"]
                offsets = data["offsets"].detach().cpu().numpy()
                start_idx = data["start_idx"].to(DEVICE)
                end_idx = data["end_idx"].to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    start_logits, end_logits = model(ids, masks)
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * len(ids)

                start_idx_np = start_idx.detach().cpu().numpy()
                end_idx_np = end_idx.detach().cpu().numpy()
                start_logits_np = (
                    torch.softmax(start_logits, dim=1).detach().cpu().numpy()
                )
                end_logits_np = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

                for i in range(len(ids)):
                    epoch_jaccard += compute_jaccard_score(
                        tweet[i],
                        start_idx_np[i],
                        end_idx_np[i],
                        start_logits_np[i],
                        end_logits_np[i],
                        offsets[i],
                    )

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

            print(
                "Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}".format(
                    epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard
                )
            )

    torch.save(model.state_dict(), filename)




## === cell 8
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)



## === cell 9
DO_TRAIN = False

if DO_TRAIN:
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df.sentiment), start=1
    ):
        print(f"Fold: {fold}")
        model = TweetModel()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999)
        )
        criterion = loss_fn
        dataloaders_dict = get_train_val_loaders(
            train_df, train_idx, val_idx, BATCH_SIZE
        )
        train_model(
            model,
            dataloaders_dict,
            criterion,
            optimizer,
            EPOCHS,
            f"roberta_fold{fold}.pth",
        )




## === cell 10
def _valid_offset_span(offsets_1d_np: np.ndarray):
    nonzero = np.where(offsets_1d_np[:, 1] > offsets_1d_np[:, 0])[0]
    if len(nonzero) == 0:
        return 0, len(offsets_1d_np) - 1
    return int(nonzero.min()), int(nonzero.max())


test_df = pd.read_csv(TEST_PATH)
test_df["text"] = test_df["text"].astype(str)

test_loader = get_test_loader(test_df, batch_size=BATCH_SIZE)

predictions = []
models = []

torch.set_grad_enabled(False)

for fold in range(skf.n_splits):
    weight_path = os.path.join(ROBERTAFOLD, f"roberta_fold{fold+1}.pth")
    if os.path.exists(weight_path):
        model = TweetModel().to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.eval()
        models.append(model)

if len(models) == 0:
    model = TweetModel().to(DEVICE)
    model.eval()
    models = [model]

for data in test_loader:
    ids = data["ids"].to(DEVICE, non_blocking=True)
    masks = data["masks"].to(DEVICE, non_blocking=True)
    tweet = data["tweet"]
    offsets = data["offsets"].detach().cpu().numpy()

    start_logits_list = []
    end_logits_list = []

    for model in models:
        output = model(ids, masks)
        start_logits_list.append(torch.softmax(output[0], dim=1).detach().cpu().numpy())
        end_logits_list.append(torch.softmax(output[1], dim=1).detach().cpu().numpy())

    start_logits = np.mean(start_logits_list, axis=0)
    end_logits = np.mean(end_logits_list, axis=0)

    for i in range(len(ids)):
        s_min, s_max = _valid_offset_span(offsets[i])

        start_pred = int(np.argmax(start_logits[i]))
        end_pred = int(np.argmax(end_logits[i]))

        start_pred = max(s_min, min(start_pred, s_max))
        end_pred = max(s_min, min(end_pred, s_max))

        if start_pred > end_pred:
            pred = tweet[i]
        else:
            pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
        predictions.append(pred)



## === cell 11
sub_df = pd.read_csv(SAMPLE_SUB_PATH)

if len(predictions) != len(sub_df):
    if len(predictions) > len(sub_df):
        predictions = predictions[: len(sub_df)]
    else:
        predictions = predictions + [""] * (len(sub_df) - len(predictions))

sub_df["selected_text"] = predictions

sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(str(x).split()) == 1 else x
)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
