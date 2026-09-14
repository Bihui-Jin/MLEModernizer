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

0.4356454908847809

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the environment-breaking import error by removing the unused TensorFlow/Keras pieces and forcing a compatible `transformers` import path, which resolves the protobuf `MessageFactory` issue. I also fix tokenizer/model loading by using the local Kaggle BERT directory (or a safe fallback to `bert-base-uncased` if the local folder isn’t present) instead of passing a `vocab.txt` path that newer Hugging Face validators reject. Next, I ensure `num_of_val` is reset per sentiment loop (it was being accumulated across categories, breaking shapes) and make the prediction-to-text function robust by converting tensors to numpy and handling missing tokens safely. Finally, I generate the submission via a vectorized merge on `textID` (no per-row append), guaranteeing the output has the required columns and writes a valid `submission.csv`.'

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


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/1168638290.py in <cell line: 0>()
     15 # Also import transformers in a minimal, compatible way.
     16 import transformers
---> 17 from transformers import AdamW
     18 
     19 

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
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
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
def from_predicted_positon_to_text(
    predicted, threshold, padded_tokens, tokens_dict, raw_text
):
    if torch.is_tensor(predicted):
        predicted = predicted.detach().float().cpu().numpy()
    if torch.is_tensor(padded_tokens):
        padded_tokens = padded_tokens.detach().cpu().numpy()

    predicted = predicted.copy()
    predicted[predicted >= threshold] = 1.0
    predicted[predicted < threshold] = 0.0

    index_matrix = []
    for i in range(len(predicted)):
        index = [ix for ix, p in enumerate(predicted[i, :].tolist()) if p == 1.0]
        index_matrix.append(index)

    first_last_word_index = []
    for token_row, idxs in zip(padded_tokens.tolist(), index_matrix):
        if len(idxs) == 0:
            first_last_word_index.append([-1, -1])
            continue

        last = len(idxs) - 1
        while last >= 0 and token_row[idxs[last]] == 0:
            last -= 1
        if last < 0:
            first_last_word_index.append([-1, -1])
        else:
            first_last_word_index.append([token_row[idxs[0]], token_row[idxs[last]]])

    inv_tokens = {}
    for k, v in tokens_dict.items():
        if v not in inv_tokens:
            inv_tokens[v] = k

    predicted_val_text = []
    for (f_id, l_id), text in zip(first_last_word_index, raw_text):
        text = str(text)
        if f_id == -1 or l_id == -1:
            predicted_val_text.append(text)
            continue

        f_tok = inv_tokens.get(f_id, None)
        l_tok = inv_tokens.get(l_id, None)
        if not f_tok or not l_tok:
            predicted_val_text.append(text)
            continue

        try:
            f_index = text.index(f_tok)
            l_index = text.index(l_tok) + len(l_tok)
            p_text = text[f_index:l_index]
        except ValueError:
            p_text = text
        predicted_val_text.append(p_text)

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
        self.linear_model = torch.nn.Linear(768, outputSize)
        torch.nn.init.normal_(self.linear_model.weight, std=std)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        _, _, hidden_states = self.bert_model(
            input_ids, attention_mask=attention_mask.float()
        )
        attention_hidden_states = hidden_states[1:]
        summed_last_4_layers = torch.stack(attention_hidden_states[-4:]).sum(0)
        sentence_embedding = torch.mean(summed_last_4_layers, dim=1)

        out = self.drop_out(sentence_embedding)
        out = self.linear_model(out)
        out = self.sigmoid(out)
        return out




## === cell 13
torch.cuda.empty_cache()

category = ["positive", "neutral", "negative"]
color = ["b", "r", "g"]
max_sequence_length = 32
val_frac = 0.25
num_of_val_base = 400  # Fix: don't mutate across categories
learningRate = 3e-4
threshold = 0.80
max_length = max_sequence_length
outputSize = max_sequence_length
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = [10, 3, 9]
predicted_text = {}

LOCAL_BERT_DIR = "/kaggle/input/bertbaseuncased"
UNCASED = LOCAL_BERT_DIR if os.path.isdir(LOCAL_BERT_DIR) else "bert-base-uncased"

std = 0.02
droupout = 0.1
batch_size = num_of_val_base
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

tokens_dict = {}

for i in range(len(train_all)):
    tokenizer = transformers.BertTokenizer.from_pretrained(UNCASED)

    tokens_dict = {}
    for k, v in list(tokenizer.vocab.items()):
        tok = k
        if tok in ["[SEP]", "[MASK]", "[CLS]", "[PAD]"]:
            continue
        if tok.startswith("##"):
            tok = tok[2:]
        tokens_dict[tok] = v

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
        for t in X_test.text.tolist()
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
        sel_ids = set(y_train_all_input_ids[j, :].tolist())
        a = [
            1 if (x > 0 and x in sel_ids) else 0
            for x in X_train_all_input_ids[j, :].tolist()
        ]
        y_train_bool.append(a)
    y_train_bool = np.array(y_train_bool, dtype=np.int64)

    num_of_val = num_of_val_base
    num_of_val = num_of_val + (len(X_train_all_input_ids) - num_of_val) % batch_size
    num_of_train = len(X_train_all_input_ids) - num_of_val

    X_val = X_train_all_input_ids[-num_of_val:, :]
    X_val_attention_mask = X_train_all_attention_mask[-num_of_val:, :]
    y_val = y_train_bool[-num_of_val:, :]

    optimizer.zero_grad()
    training_loss = []

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

            if k_fold % 1 == 0:
                print(
                    "{}-th category, total {} fold, now {}, epoch {}, loss {}".format(
                        i, int(num_of_train / batch_size), k_fold, epoch, loss.item()
                    )
                )

    plt.plot(range(len(training_loss)), training_loss, color[i], label="training loss")

    with torch.no_grad():
        X_val_t = Variable(torch.from_numpy(X_val).to(device))
        X_val_attention_mask_t = Variable(
            torch.from_numpy(X_val_attention_mask).to(device)
        )
        eval_predicted = model(X_val_t, X_val_attention_mask_t)

    predicted_val_text = from_predicted_positon_to_text(
        eval_predicted, threshold, X_val, tokens_dict, X_train_text[-num_of_val:]
    )
    jaccard_score_list = [
        jaccard(str1, str2)
        for str1, str2 in zip(predicted_val_text, y_train_text[-num_of_val:])
    ]
    result = pd.Series(jaccard_score_list)
    print(result.describe())

    with torch.no_grad():
        X_test_input_ids_t = Variable(torch.from_numpy(X_test_input_ids).to(device))
        X_test_attention_mask_t = Variable(
            torch.from_numpy(X_test_attention_mask).to(device)
        )
        test_predicted = model(X_test_input_ids_t, X_test_attention_mask_t)

    predicted_test_text = from_predicted_positon_to_text(
        test_predicted, threshold, X_test_input_ids, tokens_dict, X_test.text.tolist()
    )
    for p, idx in zip(predicted_test_text, X_test.textID.tolist()):
        predicted_text[idx] = p



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
pred_df = pd.DataFrame(
    {
        "textID": list(predicted_text.keys()),
        "selected_text": list(predicted_text.values()),
    }
)
submission = sample_submission[["textID"]].merge(pred_df, on="textID", how="left")

submission["selected_text"] = submission["selected_text"].fillna("")

submission = submission[["textID", "selected_text"]]
submission.head()



## === cell 15
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.columns.tolist())
