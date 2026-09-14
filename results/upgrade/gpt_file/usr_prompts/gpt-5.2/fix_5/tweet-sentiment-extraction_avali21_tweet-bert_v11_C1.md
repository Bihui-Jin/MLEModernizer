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

0.4605951011180877

# 6. Current score

0.59324

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'We fix the environment-breaking import error by removing unused `tensorflow` and `nltk` imports that trigger the protobuf `MessageFactory` issue. We also fix the HuggingFace tokenizer/model loading by switching `from_pretrained(VOCAB)` to `from_pretrained(UNCASED, local_files_only=True)` and adding a safe fallback to the public `bert-base-uncased` if the Kaggle dataset path isn’t present. Finally, we make the text reconstruction robust (avoid regex index errors and handle empty selections), and we generate the submission by merging on `textID` (no deprecated `append`, no missing IDs), ensuring the output CSV has exactly `textID,selected_text`.'
- What this solution (achieved 0.59324) has done: 'I fix the immediate runtime break caused by `transformers.AdamW` being removed in newer Transformers by switching to `torch.optim.AdamW` (same optimizer semantics for this use). Then I address the protobuf `MessageFactory.GetPrototype` crash by forcing Transformers to avoid importing the fast tokenizers stack (which triggers that protobuf path in some Kaggle images) and by preferring `BertTokenizer` with `use_fast=False`. Since your current score (0.59324) is already well above the target (0.4606) and within the ±10% tolerance band, I not change thresholds/training logic to avoid drifting score away from the target; all changes are stability-only. The script still write a valid `submission.csv` with exactly `textID,selected_text`.'
- What this solution (achieved 0.59324) has done: 'We fix the runtime crash (`MessageFactory` has no `GetPrototype`) by forcing Transformers to avoid importing the tokenizers/protobuf stack that triggers it: disable fast tokenizers explicitly and also set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing `transformers`. This is a stability-only change and should not materially alter model logic or training semantics. We also make the BERT local path more robust by checking the typical Kaggle-mounted directory and falling back to `bert-base-uncased` without `local_files_only` if needed. Finally, we ensure submission generation always fills every `textID` and writes a valid `submission.csv`.'
- What this solution (achieved 0.59324) has done: 'We fix the `MessageFactory.GetPrototype` crash by preventing protobuf C++ bindings from loading (force pure-Python protobuf) and by setting the related env vars *before* any library import that may indirectly touch protobuf/transformers. We also harden the Transformers imports by avoiding the top-level `import transformers` (which can trigger optional fast-tokenizer/protobuf paths) and instead importing only the specific BERT classes we use. These are stability-only changes and should not materially change model behavior/score; since your current score (0.59324) is already well above the target band, we won’t change thresholds/training logic. Finally, we keep submission writing identical but ensure it always produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import string
import math
import copy

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

os.environ["TRANSFORMERS_NO_FAST_TOKENIZERS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

from transformers import BertTokenizer, BertModel as HFBertModel, BertConfig

from torch.optim import AdamW  # transformers.AdamW removed in newer transformers



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
sample_submission = pd.read_csv(
    "../input/tweet-sentiment-extraction/sample_submission.csv"
)



## === cell 2
train.dropna(inplace=True)




## === cell 3
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text




## === cell 4
train["text_raw"] = train["text"]
train["selected_text_raw"] = train["selected_text"]
test["text_raw"] = test["text"]

train["text"] = train["text"].apply(clean_text)
train["selected_text"] = train["selected_text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)



## === cell 5
train_positive = train.loc[train.sentiment == "positive"]
train_neutral = train.loc[train.sentiment == "neutral"]
train_negative = train.loc[train.sentiment == "negative"]

test_positive = test.loc[test.sentiment == "positive"]
test_neutral = test.loc[test.sentiment == "neutral"]
test_negative = test.loc[test.sentiment == "negative"]



## === cell 6
train_all = [train_positive, train_neutral, train_negative]
test_all = [test_positive, test_neutral, test_negative]




## === cell 7
def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    if denom == 0:
        return 1.0
    return float(len(c)) / denom




## === cell 8
def from_predicted_positon_to_text(
    predicted, threshold, padded_tokens, raw_text, tokenizer
):
    """
    Robust version:
    - avoids regex failures when first/last tokens are empty or not found
    - falls back to raw_text when nothing selected
    """
    if torch.is_tensor(predicted):
        predicted = predicted.detach().cpu().numpy()

    predicted = predicted.copy()
    predicted[predicted >= threshold] = 1
    predicted[predicted < threshold] = 0

    tokens_matrix = []
    decode_matrix = []
    for i in range(padded_tokens.shape[0]):
        decode = tokenizer.decode(padded_tokens[i], skip_special_tokens=False)
        toks = tokenizer.tokenize(decode)
        tokens_matrix.append(toks)
        decode_matrix.append(" ".join(toks))

    index_matrix = []
    for i in range(len(predicted)):
        idxs = [k for k, p in enumerate(predicted[i, :].tolist()) if p == 1]
        index_matrix.append(idxs)

    predicted_val_text = []
    for toks, idxs, dec, rt in zip(
        tokens_matrix, index_matrix, decode_matrix, raw_text
    ):
        idxs = [k for k in idxs if k != 0]

        cleaned = []
        for k in idxs:
            if k < 0 or k >= len(toks):
                continue
            if toks[k] in ["[SEP]", "[PAD]"]:
                continue
            cleaned.append(k)

        if len(cleaned) == 0:
            predicted_val_text.append(str(rt).strip())
            continue

        first = cleaned[0]
        last = cleaned[-1]

        span_tokens = toks[first : last + 1]
        span_text = tokenizer.convert_tokens_to_string(span_tokens)

        span_text = (
            span_text.replace("[CLS]", "")
            .replace("[SEP]", "")
            .replace("[PAD]", "")
            .replace(" ##", "")
            .strip()
        )

        if span_text == "":
            span_text = str(rt).strip()

        predicted_val_text.append(span_text)

    return predicted_val_text




## === cell 9
class BertModel(torch.nn.Module):
    def __init__(self, UNCASED, outputSize, droupout, std):
        super(BertModel, self).__init__()
        self.config = BertConfig.from_pretrained(
            UNCASED, output_hidden_states=True, local_files_only=True
        )
        self.bert_model = HFBertModel.from_pretrained(
            UNCASED, config=self.config, local_files_only=True
        )
        self.drop_out = torch.nn.Dropout(droupout)
        self.con_model1 = torch.nn.Conv1d(
            in_channels=768, out_channels=256, kernel_size=1
        )
        self.con_model2 = torch.nn.Conv1d(
            in_channels=256, out_channels=64, kernel_size=1
        )
        self.con_model3 = torch.nn.Conv1d(
            in_channels=64, out_channels=outputSize, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        last_hidden_states = self.bert_model(
            input_ids, attention_mask=attention_mask.float()
        )
        last_hidden_states = last_hidden_states[0].permute(0, 2, 1)
        out = self.drop_out(last_hidden_states)
        out = self.con_model1(out)
        out = self.con_model2(out)
        out = self.con_model3(out)
        out = torch.sum(out, dim=2)
        out = self.sigmoid(out)
        return out




## === cell 10
torch.cuda.empty_cache()

category = ["positive", "neutral", "negative"]
color = ["b", "r", "g"]
max_sequence_length = 32
val_frac = 0.25
num_of_val = 1000
learningRate = 3e-5
max_length = max_sequence_length
outputSize = max_sequence_length
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = [5, 1, 6]
predicted_text = {}

UNCASED = "/kaggle/input/bertbaseuncased"
FALLBACK_UNCASED = "bert-base-uncased"

std = 0.02
droupout = 0.1
batch_size = 16
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def get_tokenizer_and_model_dir():
    if os.path.isdir(UNCASED):
        return UNCASED, True
    return FALLBACK_UNCASED, False


MODEL_DIR, MODEL_IS_LOCAL = get_tokenizer_and_model_dir()



## === cell 11
for i in range(len(train_all)):
    threshold = 0 if i == 1 else 0.5

    try:
        tokenizer = BertTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=MODEL_IS_LOCAL, use_fast=False
        )
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

    try:
        model = BertModel(MODEL_DIR, outputSize, droupout, std)
    except Exception:

        class BertModelFallback(BertModel):
            def __init__(self, UNCASED, outputSize, droupout, std):
                super(torch.nn.Module, self).__init__()
                self.config = BertConfig.from_pretrained(
                    UNCASED, output_hidden_states=True
                )
                self.bert_model = HFBertModel.from_pretrained(
                    UNCASED, config=self.config
                )
                self.drop_out = torch.nn.Dropout(droupout)
                self.con_model1 = torch.nn.Conv1d(
                    in_channels=768, out_channels=256, kernel_size=1
                )
                self.con_model2 = torch.nn.Conv1d(
                    in_channels=256, out_channels=64, kernel_size=1
                )
                self.con_model3 = torch.nn.Conv1d(
                    in_channels=64, out_channels=outputSize, kernel_size=1
                )
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, input_ids, attention_mask):
                last_hidden_states = self.bert_model(
                    input_ids, attention_mask=attention_mask.float()
                )
                last_hidden_states = last_hidden_states[0].permute(0, 2, 1)
                out = self.drop_out(last_hidden_states)
                out = self.con_model1(out)
                out = self.con_model2(out)
                out = self.con_model3(out)
                out = torch.sum(out, dim=2)
                out = self.sigmoid(out)
                return out

        model = BertModelFallback(MODEL_DIR, outputSize, droupout, std)

    model.to(device)

    criterion = torch.nn.MSELoss()
    param_optimizer = list(model.named_parameters())
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_parameters, lr=learningRate)

    train_data = train_all[i]
    test_data = test_all[i]

    X_train_text_raw = train_data["text_raw"].tolist()
    y_train_text_raw = train_data["selected_text_raw"].tolist()
    X_test_text_raw = test_data["text_raw"].tolist()

    X_train_text = train_data["text"].tolist()
    y_train_text = train_data["selected_text"].tolist()

    X_test = test_data[["textID", "text"]]

    X_train_tokens = [
        tokenizer.encode(
            t, add_special_tokens=True, max_length=max_length, truncation=True
        )
        for t in X_train_text
    ]
    y_train_tokens = [
        tokenizer.encode(
            t, add_special_tokens=True, max_length=max_length, truncation=True
        )
        for t in y_train_text
    ]
    X_test_tokens = [
        tokenizer.encode(
            t, add_special_tokens=True, max_length=max_length, truncation=True
        )
        for t in X_test["text"].tolist()
    ]

    X_train_all_input_ids = np.array(
        [tok + [0] * (max_length - len(tok)) for tok in X_train_tokens], dtype=np.int64
    )
    y_train_all_input_ids = np.array(
        [tok + [0] * (max_length - len(tok)) for tok in y_train_tokens], dtype=np.int64
    )
    X_test_input_ids = np.array(
        [tok + [0] * (max_length - len(tok)) for tok in X_test_tokens], dtype=np.int64
    )

    X_train_all_attention_mask = np.where(X_train_all_input_ids != 0, 1, 0).astype(
        np.int64
    )
    X_test_attention_mask = np.where(X_test_input_ids != 0, 1, 0).astype(np.int64)

    y_train_bool = []
    for j in range(len(y_train_all_input_ids)):
        a = [
            (
                1
                if (
                    x > 0 and x in y_train_all_input_ids[j, :] and (x not in [101, 102])
                )
                else 0
            )
            for x in X_train_all_input_ids[j, :].tolist()
        ]
        y_train_bool.append(a)
    y_train_bool = np.array(y_train_bool, dtype=np.int64)

    local_num_of_val = min(num_of_val, len(X_train_all_input_ids) - batch_size)
    local_num_of_val = (
        local_num_of_val + (len(X_train_all_input_ids) - local_num_of_val) % batch_size
    )
    num_of_train = len(X_train_all_input_ids) - local_num_of_val
    num_of_train = (num_of_train // batch_size) * batch_size  # ensure full batches

    X_val = X_train_all_input_ids[-local_num_of_val:, :]
    X_val_attention_mask = X_train_all_attention_mask[-local_num_of_val:, :]
    y_val = y_train_bool[-local_num_of_val:, :]

    training_loss = []

    model.train()
    for epoch in range(epochs[i]):
        for k_fold in range(int(num_of_train / batch_size)):
            X_train_batch = X_train_all_input_ids[
                k_fold * batch_size : (k_fold + 1) * batch_size, :
            ]
            X_train_attention_mask_batch = X_train_all_attention_mask[
                k_fold * batch_size : (k_fold + 1) * batch_size, :
            ]
            y_train_batch = y_train_bool[
                k_fold * batch_size : (k_fold + 1) * batch_size, :
            ]

            X_train_batch = Variable(torch.from_numpy(X_train_batch).to(device))
            X_train_attention_mask_batch = Variable(
                torch.from_numpy(X_train_attention_mask_batch).to(device)
            )
            y_train_batch = Variable(torch.from_numpy(y_train_batch).to(device))

            outputs = model(X_train_batch, X_train_attention_mask_batch)
            loss = criterion(outputs.float(), y_train_batch.float())
            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if k_fold % 100 == 0:
                print(
                    f"{i}-th category, total {int(num_of_train/batch_size)} fold, now {k_fold}, epoch {epoch}, loss {loss.item()}"
                )

    model.eval()
    with torch.no_grad():
        X_val_t = Variable(torch.from_numpy(X_val).to(device))
        X_val_mask_t = Variable(torch.from_numpy(X_val_attention_mask).to(device))
        eval_predicted = model(X_val_t, X_val_mask_t)

    predicted_val_text = from_predicted_positon_to_text(
        eval_predicted,
        threshold,
        X_train_all_input_ids[-local_num_of_val:, :],
        X_train_text_raw[-local_num_of_val:],
        tokenizer,
    )

    jaccard_score_list = [
        jaccard(p, y)
        for p, y in zip(predicted_val_text, y_train_text[-local_num_of_val:])
    ]
    result = pd.Series(jaccard_score_list)
    print(result.describe())

    with torch.no_grad():
        X_test_input_ids_t = Variable(torch.from_numpy(X_test_input_ids).to(device))
        X_test_mask_t = Variable(torch.from_numpy(X_test_attention_mask).to(device))
        test_predicted = model(X_test_input_ids_t, X_test_mask_t)

    predicted_test_text = from_predicted_positon_to_text(
        test_predicted, threshold, X_test_input_ids, X_test_text_raw, tokenizer
    )

    for p, idx in zip(predicted_test_text, X_test["textID"].tolist()):
        predicted_text[idx] = p



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1614134020.py in <cell line: 0>()
     11     try:
---> 12         model = BertModel(MODEL_DIR, outputSize, droupout, std)
     13     except Exception:

/tmp/ipykernel_55/1926070132.py in __init__(self, UNCASED, outputSize, droupout, std)
      6         )
----> 7         self.bert_model = HFBertModel.from_pretrained(
      8             UNCASED, config=self.config, local_files_only=True

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in _wrapper(*args, **kwargs)
    310         try:
--> 311             return func(*args, **kwargs)
    312         finally:

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, weights_only, *model_args, **kwargs)
   4707             and not is_sharded
-> 4708             and checkpoint_files[0].endswith(".safetensors")
   4709         ):

AttributeError: 'NoneType' object has no attribute 'endswith'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1614134020.py in <cell line: 0>()
     47                 return out
     48 
---> 49         model = BertModelFallback(MODEL_DIR, outputSize, droupout, std)
     50 
     51     model.to(device)

/tmp/ipykernel_55/1614134020.py in __init__(self, UNCASED, outputSize, droupout, std)
     19                     UNCASED, output_hidden_states=True
     20                 )
---> 21                 self.bert_model = HFBertModel.from_pretrained(
     22                     UNCASED, config=self.config
     23                 )

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __setattr__(self, name, value)
   1963             if isinstance(value, Module):
   1964                 if modules is None:
-> 1965                     raise AttributeError(
   1966                         "cannot assign module before Module.__init__() call"
   1967                     )

AttributeError: cannot assign module before Module.__init__() call

## === cell 12
submission = sample_submission.copy()
submission["selected_text"] = submission["textID"].map(predicted_text)

missing_mask = submission["selected_text"].isna()
if missing_mask.any():
    test_text_map = dict(zip(test["textID"], test["text_raw"]))
    submission.loc[missing_mask, "selected_text"] = (
        submission.loc[missing_mask, "textID"].map(test_text_map).fillna("")
    )

submission["selected_text"] = submission["selected_text"].astype(str).str.strip()

submission = submission[["textID", "selected_text"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
