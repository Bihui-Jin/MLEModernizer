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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.14

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.9720497852791602

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from collections import Counter
import re
import gc
from pathlib import Path


class Config:
    MAX_LEN = 200  # 文章の長さ
    MAX_FEATURES = 20000  # 頻出2万語を使用
    EMBED_DIM = 128  # 単語ベクトルのサイズ
    HIDDEN_DIM = 64  # GRUの記憶容量
    BATCH_SIZE = 64  # バッチサイズ
    EPOCHS = 3  # 学習回数（少し増やしてスコア向上を狙う）
    LR = 0.001  # 学習率


print("【GRU】データを読み込んでいます...")
base_path = Path("data/jigsaw-toxic-comment-classification-challenge")
train_path = base_path / "train.csv.zip"
test_path = base_path / "test.csv.zip"
sample_path = base_path / "sample_submission.csv.zip"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

train_df["comment_text"] = train_df["comment_text"].fillna("fillna")
test_df["comment_text"] = test_df["comment_text"].fillna("fillna")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2652349759.py in <cell line: 0>()
     28 sample_path = base_path / "sample_submission.csv.zip"
     29 
---> 30 train_df = pd.read_csv(train_path)
     31 test_df = pd.read_csv(test_path)
     32 sub = pd.read_csv(sample_path)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    792             # "Union[str, BaseBuffer]"; expected "Union[Union[str, PathLike[str]],
    793             # ReadBuffer[bytes], WriteBuffer[bytes]]"
--> 794             handle = _BytesZipFile(
    795                 handle, ioargs.mode, **compression_args  # type: ignore[arg-type]
    796             )

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in __init__(self, file, mode, archive_name, **kwargs)
   1035         # error: Incompatible types in assignment (expression has type "ZipFile",
   1036         # base class "_BufferedWriter" defined the type as "BytesIO")
-> 1037         self.buffer: zipfile.ZipFile = zipfile.ZipFile(  # type: ignore[assignment]
   1038             file, mode, **kwargs
   1039         )

/usr/lib/python3.11/zipfile.py in __init__(self, file, mode, compression, allowZip64, compresslevel, strict_timestamps, metadata_encoding)
   1293             while True:
   1294                 try:
-> 1295                     self.fp = io.open(file, filemode)
   1296                 except OSError:
   1297                     if filemode in modeDict:

FileNotFoundError: [Errno 2] No such file or directory: 'data/jigsaw-toxic-comment-classification-challenge/train.csv.zip'

## === cell 1
print("前処理（URL削除など）を実行中...")


def clean_for_transformer(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # URL
    text = re.sub(r"<.*?>", "", text)  # HTMLタグ
    text = re.sub(r"\n+", " ", text)  # 改行
    text = re.sub(r"\t+", " ", text)  # タブ
    text = re.sub(r" {2,}", " ", text).strip()  # 連続スペース
    return text


train_df["comment_text"] = train_df["comment_text"].apply(clean_for_transformer)
test_df["comment_text"] = test_df["comment_text"].apply(clean_for_transformer)


def simple_clean(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text.lower().split()


print("辞書を作成中...")
all_words = []
for text in train_df["comment_text"][:50000]:
    all_words.extend(simple_clean(text))

word_counts = Counter(all_words)
vocab = {
    word: i + 1
    for i, (word, _) in enumerate(word_counts.most_common(Config.MAX_FEATURES))
}
vocab_size = len(vocab) + 1  # +1 for padding index 0


def text_to_sequence(text, max_len):
    words = simple_clean(text)
    seq = [vocab.get(w, 0) for w in words]
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq


print("テキストをベクトル化中...")
X = np.array([text_to_sequence(t, Config.MAX_LEN) for t in train_df["comment_text"]])
X_test = np.array(
    [text_to_sequence(t, Config.MAX_LEN) for t in test_df["comment_text"]]
)
y = train_df[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/193585552.py in <cell line: 0>()
     13 
     14 
---> 15 train_df["comment_text"] = train_df["comment_text"].apply(clean_for_transformer)
     16 test_df["comment_text"] = test_df["comment_text"].apply(clean_for_transformer)
     17 

NameError: name 'train_df' is not defined

## === cell 2
class ToxicDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float) if y is not None else None

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]

    def __len__(self):
        return len(self.x)


train_loader = DataLoader(
    ToxicDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(ToxicDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)
test_loader = DataLoader(ToxicDataset(X_test), batch_size=Config.BATCH_SIZE)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2349645834.py in <cell line: 0>()
     14 
     15 train_loader = DataLoader(
---> 16     ToxicDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True
     17 )
     18 val_loader = DataLoader(ToxicDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)

NameError: name 'X_train' is not defined

## === cell 3
class BiGRU_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, Config.EMBED_DIM, padding_idx=0)
        self.gru = nn.GRU(
            Config.EMBED_DIM,
            Config.HIDDEN_DIM,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(Config.HIDDEN_DIM * 2, 6)

    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.gru(x)
        out, _ = torch.max(gru_out, dim=1)
        out = self.fc(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiGRU_Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
criterion = nn.BCEWithLogitsLoss()

print(f"学習を開始します (Device: {device})")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3086671021.py in <cell line: 0>()
     22 
     23 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
---> 24 model = BiGRU_Model().to(device)
     25 optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
     26 criterion = nn.BCEWithLogitsLoss()

/tmp/ipykernel_55/3086671021.py in __init__(self)
      2     def __init__(self):
      3         super().__init__()
----> 4         self.embedding = nn.Embedding(vocab_size, Config.EMBED_DIM, padding_idx=0)
      5         self.gru = nn.GRU(
      6             Config.EMBED_DIM,

NameError: name 'vocab_size' is not defined

## === cell 4
for epoch in range(Config.EPOCHS):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            out = torch.sigmoid(model(x_batch)).cpu().numpy()
            val_preds.append(out)
            val_targets.append(y_batch.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    try:
        val_auc = roc_auc_score(val_targets, val_preds, average="macro")
    except ValueError:
        val_auc = float("nan")
    print(f"Epoch {epoch+1}/{Config.EPOCHS} 完了 - Validation macro AUC: {val_auc:.5f}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4052902856.py in <cell line: 0>()
      1 for epoch in range(Config.EPOCHS):
----> 2     model.train()
      3     for x_batch, y_batch in train_loader:
      4         x_batch = x_batch.to(device)
      5         y_batch = y_batch.to(device)

NameError: name 'model' is not defined

## === cell 5
print("予測を実行中...")
model.eval()
preds = []
with torch.no_grad():
    for x_batch in test_loader:
        x_batch = x_batch.to(device)
        out = torch.sigmoid(model(x_batch)).cpu().numpy()
        preds.append(out)

final_preds = np.concatenate(preds)
sub[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = (
    final_preds
)

output_path = "submission_gru_cleaned.csv"
sub.to_csv(output_path, index=False)
print(f"完了！ '{output_path}' を保存しました。")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2763236522.py in <cell line: 0>()
      1 print("予測を実行中...")
----> 2 model.eval()
      3 preds = []
      4 with torch.no_grad():
      5     for x_batch in test_loader:

NameError: name 'model' is not defined
