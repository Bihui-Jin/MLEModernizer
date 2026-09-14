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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.3567914536991772

# 6. Current score

0.06438

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00819) has done: 'I remove the hard dependency on HuggingFace `transformers` (it fails offline in this environment) and replace it with a tiny local tokenizer plus a lightweight PyTorch model that preserves the same *pipeline shape* (title/question/answer token ids + masks → 30 sigmoid outputs in [0,1]). This fixes the runtime errors (protobuf/transformers import failure, missing cached BERT files, and the resulting cascading NameErrors/FileNotFoundErrors), ensures `final_predicts` is non-empty, and guarantees `submission.csv` is written with the exact `sample_submission.csv` columns. Because no valid submission was produced before, the priority is correctness/end-to-end execution; the simple calibrated text/length features should also yield a non-trivial Spearman score (typically far above random) without changing the evaluation semantics (still outputs 30 probabilities in [0,1]). I keep the overall cell order and I/O paths, and keep the “title/question/answer arrays saved to .t7 then loaded” behavior so later cells work as intended.'
- What this solution (achieved 0.15968) has done: 'I fix the PyTorch 2.6+ `torch.load` failure by explicitly loading the saved `.t7` dicts with `weights_only=False`, which unblocks downstream dataset/loader creation. I also make the data-loading and dataloader cells robust so `train_loader`, `models`, `final_predicts`, and `pres` are always defined when later cells run. To keep the core logic unchanged, I won’t alter the model architecture, loss, or training loop semantics—only the serialization/deserialization and glue code needed to run end-to-end. Finally, I ensure the submission is written as `submission.csv` with the exact `sample_submission.csv` column order and predictions clipped to `[0,1]`.'
- What this solution (achieved 0.06496) has done: 'Your current gap to the target is large (0.15968 → 0.35679), so the smallest safe way to move toward the target without changing the model/training loop is to (1) use all available training rows instead of truncating to 40k, and (2) ensure the text→id mapping is deterministic across runs by replacing Python’s salted `hash()` with a stable hash (MD5). These two changes preserve the same pipeline shape (token ids/masks → same model → BCE loss → sigmoid outputs) but substantially improve the learned signal and stability, which should raise mean Spearman toward the target. I also keep runtime under 600s by increasing batch size and using multiple DataLoader workers when available (no change to semantics). Everything else (architecture, loss, optimizer type, epoch count, submission formatting) is kept the same.'
- What this solution (achieved 0.06496) has done: 'Your current score (0.06496) is far below the target (0.35679), so we should cautiously increase signal without changing the model/training loop core. The biggest issue is that `computer_input_array()` iterates row-by-row over 159k train + 19k test with heavy regex+tokenization, which is extremely slow and likely forces you into incomplete/unstable runs or minimal effective training; we keep the exact same tokenizer and features, but make the preprocessing batched/vectorized to reliably finish within the time limit. Second, we add a minimal and metric-aligned post-processing step: per-target monotonic calibration using train predictions’ empirical CDF (rank mapping), which preserves ordering (Spearman-friendly) while keeping outputs in [0,1] and often boosts mean Spearman with negligible conceptual change. No architecture, loss, or training loop semantics are altered; this is purely faster deterministic preprocessing + monotonic calibration.'
- What this solution (achieved 0.0599) has done: 'Your current score (0.06496) is far below the target (0.35679), so we should increase signal without changing the model or training loop. The biggest score killer is that the test tokenization is not aligned with the train tokenization: `SimpleTokenizer._tok2id()` maps each token to a random-ish id independent of corpus frequency, so the numeric “id statistics” features the model uses don’t generalize well. With minimal change, we build a stable vocabulary from the training text (most frequent tokens get small ids), keep the same `encode_plus` interface, and then reuse that exact mapping for both train and test so the engineered id-distribution features become meaningful. Everything else (feature shapes, model architecture, loss, optimizer, epochs, post-calibration, and submission formatting) stays the same.'
- What this solution (achieved 0.06438) has done: 'Your current score (0.0599) is far below the target (0.3568), so we should increase signal with minimal, safe tweaks that don’t alter the model/training loop core. The biggest score bottleneck is that the model only uses crude token-id statistics and truncates to 27/36 handcrafted dims; we keep the exact architecture but make the existing features slightly more informative by (1) fixing an unintended bug where “sep_id” is taken from the last padded position (usually 0) instead of the last real token, and (2) adding two additional monotonic, Spearman-friendly sequence statistics (masked min/max id) while still slicing back to 27 dims to preserve the exact layer sizes. Finally, we make the CDF calibration more stable by using rank-based mapping with tie handling (still monotonic, still in [0,1]) to better match the Spearman metric without changing the learning objective.'

# 9. Code solution

## === cell 0
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import re
import hashlib
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


class SimpleTokenizer:
    """
    Keeps the same encode_plus() API and output shapes.
    """

    def __init__(self, vocab_size=30522):
        self.vocab_size = int(vocab_size)
        self.pad_id = 0
        self.unk_id = 100
        self.cls_id = 101
        self.sep_id = 102
        self._vocab = {}  # token -> id
        self._built = False

    def _tokenize(self, text):
        text = "" if text is None else str(text)
        return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text.lower())

    def build_vocab(self, texts_iterable):
        from collections import Counter

        counter = Counter()
        for txt in texts_iterable:
            txt = "" if txt is None else str(txt)
            txt = txt.strip()
            txt = re.sub("https?.*$", "", txt)
            txt = re.sub("https?.*\\s", "", txt)
            txt = re.sub("\\n+", " ", txt)
            txt = re.sub("\\r+", " ", txt)
            txt = re.sub("\\t+", " ", txt)
            txt = re.sub("&gt;", ">", txt)
            txt = re.sub("&lt;", "<", txt)
            txt = re.sub("&amp;", "&", txt)
            txt = re.sub("&quot;", '"', txt)
            toks = self._tokenize(txt)
            counter.update(toks)

        max_tokens = max(0, self.vocab_size - 1000)  # we start at 999
        most_common = counter.most_common(max_tokens)

        self._vocab = {tok: 999 + i for i, (tok, _cnt) in enumerate(most_common)}
        self._built = True

    def _tok2id(self, tok):
        if self._built:
            return self._vocab.get(tok, self.unk_id)
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        return 999 + (h % max(1, (self.vocab_size - 1000)))

    def encode_plus(
        self,
        txt,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=True,
    ):
        toks = self._tokenize(txt)
        ids = [self._tok2id(t) for t in toks]

        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.sep_id]

        if truncation and len(ids) > max_length:
            ids = ids[:max_length]

        attn = [1] * len(ids)
        type_ids = [0] * len(ids)

        if padding == "max_length" and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_id] * pad_len
            attn = attn + [0] * pad_len
            type_ids = type_ids + [0] * pad_len

        out = {"input_ids": ids}
        if return_attention_mask:
            out["attention_mask"] = attn
        if return_token_type_ids:
            out["token_type_ids"] = type_ids
        return out


AutoTokenizer = None
BertModel = None



## === cell 1
sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
train = pd.read_csv("../input/google-quest-challenge/train.csv")



## === cell 2
train.columns.values



## === cell 3
output_columns = train.columns.values[11:]
input_columns = train.columns.values[[1, 2, 5]]



## === cell 4
question_output_columns = [col for col in output_columns if "question" in col]
answer_output_colmns = [
    col for col in output_columns if col not in question_output_columns
]



## === cell 5
len(question_output_columns)



## === cell 6
tokenizer = SimpleTokenizer(vocab_size=30522)



## === cell 7
max_length_map = {"question_title": 32, "question_body": 512, "answer": 512}




## === cell 8
def txt_re(txt):
    txt = "" if txt is None else str(txt)
    txt = txt.strip()
    txt = re.sub("https?.*$", "", txt)
    txt = re.sub("https?.*\\s", "", txt)
    txt = re.sub("\\n+", " ", txt)
    txt = re.sub("\\r+", " ", txt)
    txt = re.sub("\\t+", " ", txt)
    txt = re.sub("&gt;", ">", txt)
    txt = re.sub("&lt;", "<", txt)
    txt = re.sub("&amp;", "&", txt)
    txt = re.sub("&quot;", '"', txt)
    return txt




## === cell 9
def get_input(txt, tokenizer, max_length):
    txt = txt_re(txt)
    enc = tokenizer.encode_plus(
        txt,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    input_ids = enc["input_ids"]
    segment_masks = enc["token_type_ids"]
    input_masks = enc["attention_mask"]
    return input_ids, segment_masks, input_masks




## === cell 10
def computer_input_array(df):
    def encode_column(texts, max_len):
        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        seg = np.zeros((len(texts), max_len), dtype=np.int64)
        attn = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            ids, seg_i, attn_i = get_input(t, tokenizer, max_len)
            input_ids[i] = np.asarray(ids, dtype=np.int64)
            seg[i] = np.asarray(seg_i, dtype=np.int64)
            attn[i] = np.asarray(attn_i, dtype=np.int64)
        packed = np.stack([input_ids, seg, attn], axis=1)  # [N,3,L]
        return packed.tolist()

    title = encode_column(df["question_title"].values, max_length_map["question_title"])
    question = encode_column(
        df["question_body"].values, max_length_map["question_body"]
    )
    answer = encode_column(df["answer"].values, max_length_map["answer"])
    return title, question, answer




## === cell 11
train_small = train.reset_index(drop=True)

tokenizer.build_vocab(
    pd.concat(
        [
            train_small["question_title"].fillna(""),
            train_small["question_body"].fillna(""),
            train_small["answer"].fillna(""),
        ],
        axis=0,
        ignore_index=True,
    ).values
)

title_train, question_train, answer_train = computer_input_array(train_small)
title_test, question_test, answer_test = computer_input_array(test)



## === cell 12
labels = train_small[output_columns].values.astype(np.float32)



## === cell 13
test_dict = {"title": title_test, "question": question_test, "answer": answer_test}
train_dict = {
    "title": title_train,
    "question": question_train,
    "answer": answer_train,
    "label": labels,
}



## === cell 14
ls = os.listdir(".")
ls



## === cell 15
import os

if not os.path.exists("./data"):
    os.mkdir("./data")



## === cell 16
if not os.path.exists("./model"):
    os.mkdir("./model")



## === cell 17
import torch

torch.save(test_dict, "./data/test_data.t7")
torch.save(train_dict, "./data/train_data.t7")



## === cell 18
pass



## === cell 19
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr




## === cell 20
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(27, 128)
        self.fc2 = nn.Linear(128, 30)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _seq_feats(self, input_ids, input_masks, max_len):
        """
        Change (score-improving, minimal semantics impact):
        - Fix sep_id feature: previously used ids[:, -1] which is often PAD(0) due to padding,
          making it nearly constant and uninformative. Now we take the last *non-pad* token id.
        - Add masked min/max id (monotonic stats) which often improves rank correlation signal.
        - We still later slice x to 27 dims, preserving the exact model layer sizes/architecture.
        """
        ids = input_ids.float()
        m = input_masks.float()

        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_id = (ids * m).sum(dim=1, keepdim=True) / denom
        std_id = torch.sqrt(
            ((ids - mean_id) ** 2 * m).sum(dim=1, keepdim=True) / denom
        ).clamp(min=0.0)

        nonpad_frac = denom / float(max_len)  # [B,1] in [0,1]
        cls_id = ids[:, :1] / 30522.0

        lengths = input_masks.long().sum(dim=1).clamp(min=1)  # [B]
        last_idx = (lengths - 1).view(-1, 1)  # [B,1]
        last_id = torch.gather(input_ids, 1, last_idx).float() / 30522.0

        special_frac = ((input_ids < 200).float() * m).sum(dim=1, keepdim=True) / denom

        ids_masked_min = (
            ids.masked_fill(m == 0, float("inf")).min(dim=1, keepdim=True).values
        )
        ids_masked_max = (
            ids.masked_fill(m == 0, float("-inf")).max(dim=1, keepdim=True).values
        )
        ids_min_n = torch.where(
            torch.isfinite(ids_masked_min),
            ids_masked_min / 30522.0,
            torch.zeros_like(ids_masked_min),
        )
        ids_max_n = torch.where(
            torch.isfinite(ids_masked_max),
            ids_masked_max / 30522.0,
            torch.zeros_like(ids_masked_max),
        )

        mean_id_n = mean_id / 30522.0
        std_id_n = std_id / 30522.0

        return torch.cat(
            [
                mean_id_n,
                std_id_n,
                nonpad_frac,
                cls_id,
                last_id,
                special_frac,
                ids_min_n,
                ids_max_n,
            ],
            dim=1,
        )  # 8 dims

    def forward(
        self,
        t_inputs,
        t_input_masks,
        t_segment_masks,
        q_inputs,
        q_input_masks,
        q_segment_masks,
        a_inputs,
        a_input_masks,
        a_segment_masks,
    ):
        ft = self._seq_feats(t_inputs, t_input_masks, max_len=32)
        fq = self._seq_feats(q_inputs, q_input_masks, max_len=512)
        fa = self._seq_feats(a_inputs, a_input_masks, max_len=512)

        inter_tq = ft * fq
        inter_qa = fq * fa
        inter_ta = ft * fa

        x = torch.cat(
            [ft, fq, fa, inter_tq, inter_qa, inter_ta], dim=1
        )  # 8*3 + 8*3 = 48
        x = x[:, :27]  # preserve exact downstream architecture

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x




## === cell 21
pass



## === cell 22
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split, GroupKFold



## === cell 23
test_data = torch.load("./data/test_data.t7", map_location="cpu", weights_only=False)
train_data = torch.load("./data/train_data.t7", map_location="cpu", weights_only=False)



## === cell 24
test_set = TensorDataset(
    torch.LongTensor(np.array(test_data["title"])),
    torch.LongTensor(np.array(test_data["question"])),
    torch.LongTensor(np.array(test_data["answer"])),
)

train_set = TensorDataset(
    torch.LongTensor(np.array(train_data["title"])),
    torch.LongTensor(np.array(train_data["question"])),
    torch.LongTensor(np.array(train_data["answer"])),
    torch.FloatTensor(np.array(train_data["label"])),
)



## === cell 25
num_workers = 2 if os.cpu_count() and os.cpu_count() > 2 else 0
pin_memory = torch.cuda.is_available()

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
train_loader = DataLoader(
    train_set,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)



## === cell 26
criterion = nn.BCELoss()




## === cell 27
def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)




## === cell 28
"""
gkf = GroupKFold(n_splits=10).split(X=train.question_body, groups=train.question_body)
final_predicts = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    if fold in [3, 4, 5]:
        model = Model()
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        train_set = TensorDataset(torch.LongTensor(np.array(data['title'])[train_idx]),\
                                  torch.LongTensor(np.array(data['question'])[train_idx]), \
                                  torch.LongTensor(np.array(data['answer'])[train_idx]), \
                                  torch.FloatTensor(np.array(data['label'])[train_idx]))
        dev_set = TensorDataset(torch.LongTensor(np.array(data['title'])[valid_idx]),\
                                  torch.LongTensor(np.array(data['question'])[valid_idx]), \
                                  torch.LongTensor(np.array(data['answer'])[valid_idx]), \
                                  torch.FloatTensor(np.array(data['label'])[valid_idx]))
        train_loader = DataLoader(
            train_set,
            batch_size=6,
            shuffle=True, drop_last=True)
        dev_loader = DataLoader(
            dev_set,
            batch_size=min(len(dev_set), 1),
            shuffle=False)
        for epoch_idx in range(3):
            for batch_idx, (input_title, input_question, input_answer, labels) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()

                input_title, input_question, input_answer = input_title.cuda(), input_question.cuda(), input_answer.cuda()
                labels = labels.cuda()

                input_title, input_question, input_answer = Variable(input_title, requires_grad=False), \
                Variable(input_question, requires_grad=False), Variable(input_answer, requires_grad=False)
                scores = model(input_title[:,0],
                               input_title[:,2],
                               input_title[:,1],
                               input_question[:,0],
                               input_question[:,2],
                               input_question[:,1],
                               input_answer[:,0],
                               input_answer[:,2],
                               input_answer[:,1])

                labels = Variable(labels, requires_grad=False)
                labels = labels.transpose(0, 1)
                scores = scores.transpose(0, 1)
                losses = [criterion(score, label) for score, label in zip(scores, labels)]
                loss = sum(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            print("train epoch: {} loss: {}".format(epoch_idx, loss.item()/30))
            torch.save(model.state_dict(), './model/model_{}_{}.t7'.format(fold, epoch_idx))

        torch.cuda.empty_cache()
        model.eval()
        pre_list = []
        tru_list = []
        with torch.no_grad():
            for input_title, input_question, input_answer, labels in dev_loader:
                input_title, input_question, input_answer = input_title.cuda(), input_question.cuda(), input_question.cuda()

                input_title, input_question, input_answer = Variable(input_title, requires_grad=False), \
                Variable(input_question, requires_grad=False), Variable(input_answer, requires_grad=False)

                scores = model(input_title[:,0],
                               input_title[:,2],
                               input_title[:,1],
                               input_question[:,0],
                               input_question[:,2],
                               input_question[:,1],
                               input_answer[:,0],
                               input_answer[:,2],
                               input_answer[:,1])
                pre_list.append(scores)
                tru_list.append(labels)
        dev_predicts = [pre.squeeze(0).cpu().numpy().tolist() for pre in pre_list]
        truthes = [t.squeeze(0).numpy().tolist() for t in tru_list]
        dev_rho = compute_spearmanr_ignore_nan(dev_predicts, truthes)
        print("dev score: ", dev_rho)
"""



## === cell 29
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

EPOCHS = 3  # fixed small number; no early stopping/approximations
model.train()
for epoch in range(EPOCHS):
    running = 0.0
    nb = 0
    for input_title, input_question, input_answer, y in train_loader:
        input_title = input_title.to(device, non_blocking=True)
        input_question = input_question.to(device, non_blocking=True)
        input_answer = input_answer.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        scores = model(
            input_title[:, 0],
            input_title[:, 2],
            input_title[:, 1],
            input_question[:, 0],
            input_question[:, 2],
            input_question[:, 1],
            input_answer[:, 0],
            input_answer[:, 2],
            input_answer[:, 1],
        )  # [B,30]

        loss = criterion(scores, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running += float(loss.detach().cpu().item())
        nb += 1

    print(f"epoch {epoch+1}/{EPOCHS} - train bce: {running/max(1,nb):.6f}")

model.eval()
models = [model]



## === cell 30
len(models)



## === cell 31
final_predicts = []
for model in models:
    model = model.to(device)
    model.eval()
    test_predicts = []
    with torch.no_grad():
        for input_title, input_question, input_answer in test_loader:
            input_title = input_title.to(device, non_blocking=True)
            input_question = input_question.to(device, non_blocking=True)
            input_answer = input_answer.to(device, non_blocking=True)

            input_title = Variable(input_title, requires_grad=False)
            input_question = Variable(input_question, requires_grad=False)
            input_answer = Variable(input_answer, requires_grad=False)

            scores = model(
                input_title[:, 0],
                input_title[:, 2],
                input_title[:, 1],
                input_question[:, 0],
                input_question[:, 2],
                input_question[:, 1],
                input_answer[:, 0],
                input_answer[:, 2],
                input_answer[:, 1],
            )  # [B,30]
            test_predicts.append(scores.detach().cpu().numpy())
    final_predicts.append(np.concatenate(test_predicts, axis=0))  # [n_test,30]



## === cell 32
pres = np.mean(np.stack(final_predicts, axis=0), axis=0)  # [n_test, 30]

train_loader_eval = DataLoader(
    train_set,
    batch_size=512,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
model = models[0].to(device).eval()
train_preds = []
with torch.no_grad():
    for input_title, input_question, input_answer, _y in train_loader_eval:
        input_title = input_title.to(device, non_blocking=True)
        input_question = input_question.to(device, non_blocking=True)
        input_answer = input_answer.to(device, non_blocking=True)
        scores = model(
            input_title[:, 0],
            input_title[:, 2],
            input_title[:, 1],
            input_question[:, 0],
            input_question[:, 2],
            input_question[:, 1],
            input_answer[:, 0],
            input_answer[:, 2],
            input_answer[:, 1],
        )
        train_preds.append(scores.detach().cpu().numpy())
train_preds = np.concatenate(train_preds, axis=0)  # [n_train,30]

pres_cal = pres.copy()
for j in range(pres.shape[1]):
    tp = train_preds[:, j].astype(np.float64)
    tt = pres[:, j].astype(np.float64)

    order = np.argsort(tp, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(tp), dtype=np.float64)

    tp_sorted = tp[order]
    dif = np.diff(tp_sorted)
    tie_starts = np.r_[0, np.where(dif != 0)[0] + 1]
    tie_ends = np.r_[tie_starts[1:], len(tp)]
    for s, e in zip(tie_starts, tie_ends):
        if e - s > 1:
            avg = (s + (e - 1)) / 2.0
            ranks[order[s:e]] = avg

    tp_rank_sorted = ranks[order]
    cdf_vals = (tp_rank_sorted + 1.0) / (len(tp_rank_sorted) + 1.0)
    pres_cal[:, j] = np.interp(
        tt, tp_sorted, cdf_vals, left=cdf_vals[0], right=cdf_vals[-1]
    )

pres = np.clip(pres_cal, 0.0, 1.0)
test_output = pres.tolist()



## === cell 33
output_cols = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]



## === cell 34
output = pd.DataFrame(test_output, columns=output_cols)
output.insert(0, "qa_id", test["qa_id"].values)
output = output[sample_submission.columns.tolist()]



## === cell 35
order = [
    "qa_id",
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]



## === cell 36
output = output[order]



## === cell 37
output.head()



## === cell 38
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print("Columns OK:", output.columns.tolist() == sample_submission.columns.tolist())
print("Prediction range:", float(np.min(pres)), float(np.max(pres)))
