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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fix the immediate import/runtime crash by removing the unused `nltk` import that triggers the `MessageFactory.GetPrototype` protobuf issue in this environment. Then I fix the Hugging Face tokenizer/model loading by using the local BERT directory (`UNCASED`) as the `from_pretrained` path (instead of passing a vocab file path, which now fails validation), while keeping the same model architecture and training loop. Finally, I make submission creation robust by building the output from `sample_submission` directly (no deprecated `append`, no missing keys) and ensuring the CSV has exactly the required `textID,selected_text` columns.'
- What this solution (achieved 0.59324) has done: 'I fix the immediate runtime import error by switching from the removed `transformers.AdamW` to `torch.optim.AdamW` (same optimizer family, minimal change). Then I fix the Hugging Face local-loading crash by pointing `UNCASED` to a valid local BERT directory and adding a small resolver that falls back to `bert-base-uncased` if the Kaggle dataset path differs, keeping the same model and training loop. Finally, I keep the submission-writing logic but ensure `selected_text` is always a string and the CSV is written with the exact required columns and filename. These changes are score-neutral in intent (your current score is already above the target band) and primarily restore end-to-end execution.'
- What this solution (achieved 0.59324) has done: 'You’re crashing because `BertTokenizer.from_pretrained(..., local_files_only=True)` can’t find a usable local BERT vocab/config at `UNCASED`, so Transformers resolves `vocab_file=None` and then fails when checking `os.path.isfile(None)`. I fix this by making `resolve_bert_dir()` actually locate a valid local BERT folder (one that contains `vocab.txt`) by searching `/kaggle/input` and falling back to online loading only if necessary; this is a runtime fix and is score-neutral in intent. I also make the tokenizer/model loading more robust by using `AutoTokenizer`/`AutoModel` (still BERT under the hood) while preserving the same architecture, training loop, and outputs. Finally, I keep your submission-building logic but ensure every test id gets a non-null string and the CSV is written correctly.'
- What this solution (achieved 0.59324) has done: 'You’re still hitting the protobuf `MessageFactory.GetPrototype` crash because something in the import chain brings in `google.protobuf` in an incompatible way; the simplest minimal fix is to avoid importing `transformers` at module import time and instead import it inside the training cell after setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. I also make the BERT directory resolver stricter (must contain `config.json` and `pytorch_model.bin` in addition to `vocab.txt`) so `local_files_only=True` doesn’t silently pick a partial folder and then crash later. Since your current score (0.59324) is already well above the target band, I not change thresholds, epochs, model, or post-processing logic to avoid score drift; changes are intended to be runtime-stability only and still write a valid `submission.csv`.'
- What this solution (achieved 0.59324) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by ensuring the environment variable is set before any library that may import protobuf is loaded, and by deferring the `transformers` import until after that setting. I also make the BERT directory resolver slightly more permissive for local weights (accept `model.safetensors` in addition to `pytorch_model.bin`) to avoid local-loading failures that can trigger extra import paths. These changes are intended to be runtime-stability only (your current score is already above the target band), and they preserve the same model, training loop, and prediction post-processing. The script still write a valid `submission.csv` with exactly `textID,selected_text`.'
- What this solution (achieved 0.59324) has done: 'The crash is coming from an incompatible protobuf C-extension being imported somewhere in the `transformers` import chain, so we need to force protobuf to use the pure-Python implementation before **any** such imports occur (including indirect ones). I move the environment-variable setup into the very first cell and delay importing `transformers` until after that point, then import `google.protobuf` once to “lock in” the Python implementation before loading models/tokenizers. These changes are runtime-stability only (your current score is already above the target band), and they preserve the same model, training loop, thresholds, and submission logic so your score should remain essentially unchanged while producing `submission.csv` end-to-end.'
- What this solution (achieved 0.59324) has done: 'The crash comes from an incompatible protobuf backend being imported before/with `transformers`, so the environment variable alone isn’t reliably taking effect. I fix this by forcing protobuf to use the pure-Python implementation *before any protobuf import happens*, and by delaying both `google.protobuf` and `transformers` imports until after that, then “locking” the protobuf implementation. These changes are runtime-stability only (your current score is already above the target band), and they preserve the model, training loop, and post-processing so the score should not drift materially. The script then run end-to-end and write a valid `submission.csv` with exactly `textID,selected_text`.'
- What this solution (achieved 0.59324) has done: 'We need to eliminate the protobuf `MessageFactory.GetPrototype` crash by ensuring protobuf uses the pure-Python implementation before anything in the `transformers` import chain triggers the C++ backend; doing this requires setting env vars first and then explicitly importing `google.protobuf` after that, before importing `transformers`. I also make the import order stricter and add a safe fallback to avoid hard-crashing if local-only model loading fails (this is runtime-stability oriented and should be score-neutral). Since your current score (0.59324) is already above the target band, I not change training hyperparameters, thresholds, model architecture, or post-processing that would intentionally move the score. Finally, I keep submission generation identical but ensure it always writes a valid `submission.csv` with the correct columns.'
- What this solution (achieved 0.59324) has done: 'We need to eliminate the protobuf `MessageFactory.GetPrototype` crash before `transformers` is imported; setting env vars alone isn’t sufficient if protobuf was already imported/initialized. I remove the eager `google.protobuf` import and instead set env vars at the very top, then import `transformers` inside the training cell after forcing the pure-Python protobuf implementation module to be loaded first. This is a runtime-stability fix and should be score-neutral (your current score is already above the target band, so we avoid any model/training/post-processing changes). Finally, I keep submission generation the same but ensure `selected_text` is always non-null and the CSV is written as `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if os.environ.get("__PB_REEXEC_DONE__", "0") != "1":
    os.environ["__PB_REEXEC_DONE__"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.optim import AdamW



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
    text = text.lower()
    text = re.sub("https?://\\S+|www\\.\\S+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\\n", "", text)
    text = re.sub("\\w*\\d\\w*", "", text)
    return text




## === cell 4
train["text_raw"] = train["text"]
train["selected_text_raw"] = train["selected_text"]
test["text_raw"] = test["text"]

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
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    if (len(a) + len(b) - len(c)) == 0:
        return 1
    return float(len(c)) / (len(a) + len(b) - len(c))




## === cell 8
def from_predicted_positon_to_text(
    predicted, threshold, padded_tokens, raw_text, tokenizer
):
    if torch.is_tensor(predicted):
        predicted = predicted.detach().float().cpu().numpy()

    predicted = predicted.copy()
    predicted[predicted >= threshold] = 1
    predicted[predicted < threshold] = 0

    tokens_matrix = []
    decode_matrix = []
    for i in range(padded_tokens.shape[0]):
        decode = tokenizer.decode(padded_tokens[i])
        tokens_matrix.append(tokenizer.tokenize(decode))
        decode_matrix.append(" ".join(tokenizer.tokenize(decode)))

    index_matrix = []
    for i in range(len(predicted)):
        index = [ii for ii, p in enumerate(predicted[i, :].tolist()) if p == 1]
        index_matrix.append(index)

    first_last_words = []
    for toks, idxs in zip(tokens_matrix, index_matrix):
        if 0 in idxs:
            idxs.remove(0)
        for k in idxs[::-1]:
            if k < len(toks) and toks[k] in ["[SEP]", "[PAD]"]:
                idxs.remove(k)
        if len(idxs) == 0:
            first_last_words.append(["", ""])
        else:
            last = idxs[-1]
            first = idxs[0]
            if first < len(toks) and last < len(toks) and toks[first] == toks[last]:
                first_last_words.append(["", toks[last]])
            else:
                w_first = toks[first] if first < len(toks) else ""
                w_last = toks[last] if last < len(toks) else ""
                first_last_words.append([w_first, w_last])

    predicted_val_text = []
    for (w_first, w_last), s in zip(first_last_words, decode_matrix):
        try:
            if w_first == "" and w_last == "":
                pred = ""
            else:
                pred = re.findall(w_first + ".+" + w_last, s)[0]
            pred = (
                pred.replace("[CLS]", "")
                .replace("CLS]", "")
                .replace("[SEP]", "")
                .replace("[PAD]", "")
                .replace("[S", "")
                .replace("[P", "")
                .replace("AD", "")
                .replace(" ##", "")
                .strip()
            )
        except Exception:
            pred = (
                s.replace("[CLS]", "")
                .replace("[SEP]", "")
                .replace("[PAD]", "")
                .replace(" ##", "")
                .strip()
            )
        predicted_val_text.append(pred)

    return predicted_val_text




## === cell 9
def from_predicted_positon_to_text_0(
    predicted, threshold, padded_tokens, raw_text, tokenizer
):
    if torch.is_tensor(predicted):
        predicted = predicted.detach().float().cpu().numpy()

    predicted = predicted.copy()
    predicted[predicted >= threshold] = 1
    predicted[predicted < threshold] = 0

    selected_tokens = torch.from_numpy(padded_tokens).to(device) * torch.from_numpy(
        predicted
    ).to(device)

    predicted_tokens = []
    for i in range(selected_tokens.shape[0]):
        predicted_tokens.append(selected_tokens[i][selected_tokens[i] != 0])

    predicted_text = []
    for i in range(len(predicted_tokens)):
        pt = (
            tokenizer.decode(predicted_tokens[i])
            .replace("[CLS]", "")
            .replace("[SEP]", "")
        )
        predicted_text.append(pt)
    return predicted_text




## === cell 10
import transformers


class BertModel(torch.nn.Module):
    def __init__(self, UNCASED, outputSize, droupout, std):
        super(BertModel, self).__init__()
        self.config = transformers.AutoConfig.from_pretrained(
            UNCASED, output_hidden_states=True, local_files_only=True
        )
        self.bert_model = transformers.AutoModel.from_pretrained(
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


def resolve_bert_dir():
    """
    Ensure we return a complete local model directory for AutoModel/AutoTokenizer.
    Accept either pytorch_model.bin or model.safetensors, plus vocab.txt + config.json.
    """
    candidates = [
        "/kaggle/input/bertbaseuncased",
        "/kaggle/input/bert-base-uncased",
        "/kaggle/input/bertbaseuncased/bert-base-uncased",
        "/kaggle/input/bert-base-uncased/bert-base-uncased",
    ]

    def is_complete_model_dir(path):
        if not os.path.isdir(path):
            return False
        files = set(os.listdir(path))
        has_core = ("vocab.txt" in files) and ("config.json" in files)
        has_weights = ("pytorch_model.bin" in files) or ("model.safetensors" in files)
        return has_core and has_weights

    for c in candidates:
        if is_complete_model_dir(c):
            return c

    base = "/kaggle/input"
    if os.path.isdir(base):
        for root, dirs, files in os.walk(base):
            fset = set(files)
            if (
                ("vocab.txt" in fset)
                and ("config.json" in fset)
                and (("pytorch_model.bin" in fset) or ("model.safetensors" in fset))
            ):
                return root
            rel = os.path.relpath(root, base)
            if rel.count(os.sep) >= 4:
                dirs[:] = []

    return "bert-base-uncased"


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

UNCASED = resolve_bert_dir()
std = 0.02
droupout = 0.1
batch_size = 16
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

for i in range(len(train_all)):
    if i == 1:
        threshold = 0
    else:
        threshold = 0.5

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            UNCASED, local_files_only=True, use_fast=False
        )
    except Exception as e:
        print("Local tokenizer load failed, trying non-local:", repr(e))
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased", use_fast=False
        )

    try:
        model = BertModel(UNCASED, outputSize, droupout, std)
    except Exception as e:
        print("Local model load failed, trying non-local:", repr(e))
        UNCASED_FALLBACK = "bert-base-uncased"
        model = BertModel(UNCASED_FALLBACK, outputSize, droupout, std)

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

    X_train_text_raw = train_data.loc[:, "text_raw"].tolist()
    y_train_text_raw = train_data.loc[:, "selected_text_raw"].tolist()
    X_test_text_raw = test_data.loc[:, "text_raw"].tolist()

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
        [ids + [0] * (max_length - len(ids)) for ids in X_train_tokens]
    )
    y_train_all_input_ids = np.array(
        [ids + [0] * (max_length - len(ids)) for ids in y_train_tokens]
    )
    X_test_input_ids = np.array(
        [ids + [0] * (max_length - len(ids)) for ids in X_test_tokens]
    )

    X_train_all_attention_mask = np.where(X_train_all_input_ids != 0, 1, 0)
    X_test_attention_mask = np.where(X_test_input_ids != 0, 1, 0)

    y_train_bool = []
    for j in range(len(y_train_all_input_ids)):
        a = [
            (
                1
                if x > 0 and x in y_train_all_input_ids[j, :] and (x not in [101, 102])
                else 0
            )
            for x in X_train_all_input_ids[j, :].tolist()
        ]
        y_train_bool.append(a)
    y_train_bool = np.array(y_train_bool)

    num_of_val_adj = num_of_val + (len(X_train_all_input_ids) - num_of_val) % batch_size
    num_of_train = len(X_train_all_input_ids) - num_of_val_adj

    X_val = X_train_all_input_ids[-num_of_val_adj:, :]
    X_val_attention_mask = X_train_all_attention_mask[-num_of_val_adj:, :]
    y_val = y_train_bool[-num_of_val_adj:, :]

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

    plt.plot(range(len(training_loss)), training_loss, color[i], label="training loss")

    model.eval()
    with torch.no_grad():
        X_val_t = Variable(torch.from_numpy(X_val).to(device))
        X_val_attention_mask_t = Variable(
            torch.from_numpy(X_val_attention_mask).to(device)
        )
        eval_predicted = model(X_val_t, X_val_attention_mask_t)

    predicted_val_text = from_predicted_positon_to_text(
        eval_predicted,
        threshold,
        X_train_all_input_ids[-num_of_val_adj:, :],
        X_train_text_raw[-num_of_val_adj:],
        tokenizer,
    )

    jaccard_score_list = []
    for str1, str2 in zip(predicted_val_text, y_train_text[-num_of_val_adj:]):
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
        test_predicted, threshold, X_test_input_ids, X_test_text_raw, tokenizer
    )

    for p, idx in zip(predicted_test_text, X_test.textID.tolist()):
        predicted_text[idx] = p



## === cell 11
sub = sample_submission.copy()
sub["selected_text"] = sub["textID"].map(predicted_text)

fallback_map = test.set_index("textID")["text_raw"].to_dict()
sub["selected_text"] = (
    sub["selected_text"].fillna(sub["textID"].map(fallback_map)).fillna("")
)
sub["selected_text"] = sub["selected_text"].astype(str)

sub = sub[["textID", "selected_text"]]
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
