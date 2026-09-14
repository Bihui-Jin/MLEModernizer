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

0.4428064227104187

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I remove the problematic/unused imports that trigger the protobuf `MessageFactory` error and make the notebook start cleanly. Then I fix HuggingFace loading by switching `BertTokenizer.from_pretrained` to use the model directory (not the vocab file) and by using the fast, truncating tokenizer call so sequences never exceed `max_length`. Finally, I make `from_predicted_positon_to_text` robust to regex failures and build the submission in a vectorized, index-aligned way so every `textID` is filled and the output CSV has exactly the required columns.'

# 9. Code solution

## === cell 0
import os
import re
import string
import math
import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import transformers
from transformers import AdamW




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2121624248.py in <cell line: 0>()
     14 
     15 import transformers
---> 16 from transformers import AdamW
     17 
     18 # NOTE:

ImportError: cannot import name 'AdamW' from 'transformers' (/usr/local/lib/python3.11/dist-packages/transformers/__init__.py)

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
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text




## === cell 4
train["text"] = train["text"].apply(lambda x: clean_text(x))
train["selected_text"] = train["selected_text"].apply(lambda x: clean_text(x))

test["text"] = test["text"].apply(lambda x: clean_text(x))



## === cell 5
train_positive = train.loc[(train.sentiment == "positive")]
train_neutral = train.loc[(train.sentiment == "neutral")]
train_negative = train.loc[(train.sentiment == "negative")]

test_positive = test.loc[(test.sentiment == "positive")]
test_neutral = test.loc[(test.sentiment == "neutral")]
test_negative = test.loc[(test.sentiment == "negative")]



## === cell 6
train_all = [train_positive, train_neutral, train_negative]
test_all = [test_positive, test_neutral, test_negative]



## === cell 7
len(train_positive)



## === cell 8
len(train_all)



## === cell 9
test_positive.head()




## === cell 10
def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    if (len(a) + len(b) - len(c)) == 0:
        return 1.0
    return float(len(c)) / (len(a) + len(b) - len(c))




## === cell 11
def from_predicted_positon_to_text(predicted, threshold, padded_tokens, tokenizer):
    """
    Bugfixes (score-neutral but stability-critical):
    - Ensure 'predicted' is a CPU numpy array before thresholding/indexing.
    - Robustly handle cases where regex fails (no match) by falling back to decoded text.
    - Avoid relying on brittle token string cleanups.
    """
    if torch.is_tensor(predicted):
        predicted = predicted.detach().float().cpu().numpy()
    else:
        predicted = np.asarray(predicted, dtype=np.float32)

    predicted_bin = (predicted >= threshold).astype(np.int32)

    tokens_matrix = []
    decode_matrix = []
    for i in range(padded_tokens.shape[0]):
        decode = tokenizer.decode(padded_tokens[i].tolist(), skip_special_tokens=False)
        toks = tokenizer.tokenize(decode)
        tokens_matrix.append(toks)
        decode_matrix.append(" ".join(toks))

    index_matrix = []
    for i in range(len(predicted_bin)):
        index = [idx for idx, p in enumerate(predicted_bin[i, :].tolist()) if p == 1]
        index_matrix.append(index)

    first_last_words = []
    for toks, idxs in zip(tokens_matrix, index_matrix):
        if 0 in idxs:
            idxs = [x for x in idxs if x != 0]
        if len(idxs) == 0 or len(toks) == 0:
            first_last_words.append(["", ""])
        else:
            last = min(idxs[-1], len(toks) - 1)
            first = min(idxs[0], len(toks) - 1)
            if toks[first] == toks[last]:
                first_last_words.append(["", toks[last]])
            else:
                first_last_words.append([toks[first], toks[last]])

    predicted_val_text = []
    for (w1, w2), s in zip(first_last_words, decode_matrix):
        try:
            if w1 == "" and w2 == "":
                extracted = ""
            else:
                pattern = re.escape(w1) + r".+" + re.escape(w2)
                extracted = re.findall(pattern, s)[0]
        except Exception:
            extracted = s

        extracted = (
            extracted.replace("[CLS]", "")
            .replace("[SEP]", "")
            .replace("[PAD]", "")
            .replace(" ##", "")
            .strip()
        )
        predicted_val_text.append(extracted)

    return predicted_val_text




## === cell 12
class BertModel(torch.nn.Module):
    def __init__(self, UNCASED, outputSize, droupout, std):
        super(BertModel, self).__init__()
        self.config = transformers.BertConfig.from_pretrained(
            UNCASED, output_hidden_states=True
        )
        self.bert_model = transformers.BertModel.from_pretrained(
            UNCASED, config=self.config
        )
        self.drop_out = torch.nn.Dropout(droupout)
        self.con_model1 = torch.nn.Conv1d(
            in_channels=768, out_channels=256, kernel_size=1
        )
        self.con_model2 = torch.nn.Conv1d(
            in_channels=256, out_channels=outputSize, kernel_size=1
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
        out = torch.sum(out, dim=2)
        out = self.sigmoid(out)
        return out




## === cell 13
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
epochs = [3, 1, 4]
predicted_text = {}

UNCASED_LOCAL = "/kaggle/input/bertbaseuncased"
UNCASED = UNCASED_LOCAL if os.path.isdir(UNCASED_LOCAL) else "bert-base-uncased"

std = 0.02
droupout = 0.1
batch_size = 16
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
tokens_dict = {}

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

for i in range(len(train_all)):
    if i == 1:
        threshold = 0
    else:
        threshold = 0.5

    tokenizer = transformers.BertTokenizer.from_pretrained(UNCASED)

    model = BertModel(UNCASED, outputSize, droupout, std)
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

    X_train_text = train_data.loc[:, "text"].tolist()
    y_train_text = train_data.loc[:, "selected_text"].tolist()
    X_test = test_data.loc[:, ["textID", "text"]]

    X_train_tokens = [
        tokenizer.encode(
            t, add_special_tokens=True, truncation=True, max_length=max_length
        )
        for t in X_train_text
    ]
    y_train_tokens = [
        tokenizer.encode(
            t, add_special_tokens=True, truncation=True, max_length=max_length
        )
        for t in y_train_text
    ]
    X_test_tokens = [
        tokenizer.encode(
            t, add_special_tokens=True, truncation=True, max_length=max_length
        )
        for t in X_test.text.tolist()
    ]

    X_train_all_input_ids = np.array(
        [ids + [0] * (max_length - len(ids)) for ids in X_train_tokens], dtype=np.int64
    )
    y_train_all_input_ids = np.array(
        [ids + [0] * (max_length - len(ids)) for ids in y_train_tokens], dtype=np.int64
    )
    X_test_input_ids = np.array(
        [ids + [0] * (max_length - len(ids)) for ids in X_test_tokens], dtype=np.int64
    )

    X_train_all_attention_mask = np.where(X_train_all_input_ids != 0, 1, 0).astype(
        np.int64
    )
    y_train_all_attention_mask = np.where(y_train_all_input_ids != 0, 1, 0).astype(
        np.int64
    )
    X_test_attention_mask = np.where(X_test_input_ids != 0, 1, 0).astype(np.int64)

    y_train_bool = []
    for j in range(len(y_train_all_input_ids)):
        a = [
            1 if x > 0 and x in y_train_all_input_ids[j, :] else 0
            for x in X_train_all_input_ids[j, :].tolist()
        ]
        y_train_bool.append(a)
    y_train_bool = np.array(y_train_bool, dtype=np.int64)

    num_of_val_eff = min(num_of_val, len(X_train_all_input_ids) - batch_size)
    num_of_val_eff = (
        num_of_val_eff + (len(X_train_all_input_ids) - num_of_val_eff) % batch_size
    )
    num_of_train = len(X_train_all_input_ids) - num_of_val_eff

    X_val = X_train_all_input_ids[-num_of_val_eff:, :]
    X_val_attention_mask = X_train_all_attention_mask[-num_of_val_eff:, :]
    y_val = y_train_bool[-num_of_val_eff:, :]

    training_loss = []

    model.train()
    for epoch in range(epochs[i]):
        for k_fold in range(int(num_of_train / batch_size)):
            X_train_batch = X_train_all_input_ids[
                k_fold * batch_size : (k_fold + 1) * batch_size, :
            ]
            X_train_attention_batch = X_train_all_attention_mask[
                k_fold * batch_size : (k_fold + 1) * batch_size, :
            ]
            y_train_batch = y_train_bool[
                k_fold * batch_size : (k_fold + 1) * batch_size, :
            ]

            X_train_batch = Variable(torch.from_numpy(X_train_batch).to(device))
            X_train_attention_batch = Variable(
                torch.from_numpy(X_train_attention_batch).to(device)
            )
            y_train_batch = Variable(torch.from_numpy(y_train_batch).to(device))

            outputs = model(X_train_batch, X_train_attention_batch)
            loss = criterion(outputs.float(), y_train_batch.float())
            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if k_fold % 100 == 0:
                print(
                    f"{i}-th category, total {int(num_of_train/batch_size)} fold, now {k_fold}, epoch {epoch}, loss {loss.item()}"
                )

    plt.plot(range(len(training_loss)), training_loss, color[i], label="training loss")

    model.eval()
    with torch.no_grad():
        X_val_t = Variable(torch.from_numpy(X_val).to(device))
        X_val_attention_t = Variable(torch.from_numpy(X_val_attention_mask).to(device))
        eval_predicted = model(X_val_t, X_val_attention_t)

    predicted_val_text = from_predicted_positon_to_text(
        eval_predicted, threshold, X_train_all_input_ids[-num_of_val_eff:, :], tokenizer
    )

    jaccard_score_list = []
    for str1, str2 in zip(predicted_val_text, y_train_text[-num_of_val_eff:]):
        jaccard_score_list.append(jaccard(str1, str2))
    result = pd.Series(jaccard_score_list)
    print(result.describe())

    with torch.no_grad():
        X_test_input_ids_t = Variable(torch.from_numpy(X_test_input_ids).to(device))
        X_test_attention_mask_t = Variable(
            torch.from_numpy(X_test_attention_mask).to(device)
        )
        test_predicted = model(X_test_input_ids_t, X_test_attention_mask_t)

    predicted_test_text = from_predicted_positon_to_text(
        test_predicted, threshold, X_test_input_ids, tokenizer
    )
    for p, idx in zip(predicted_test_text, X_test.textID.tolist()):
        predicted_text[idx] = p



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
sub = sample_submission.copy()
sub["selected_text"] = sub["textID"].map(predicted_text).fillna("")
sub = sub[["textID", "selected_text"]]
sub.head()



## === cell 15
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.columns.tolist())
