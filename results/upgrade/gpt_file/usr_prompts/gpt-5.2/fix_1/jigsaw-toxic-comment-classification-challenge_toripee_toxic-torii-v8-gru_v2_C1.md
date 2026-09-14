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

Higher is better.

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
from collections import Counter
import re
import gc

class Config:
    MAX_LEN = 200        # 文章の長さ
    MAX_FEATURES = 20000 # 頻出2万語を使用
    EMBED_DIM = 128      # 単語ベクトルのサイズ
    HIDDEN_DIM = 64      # GRUの記憶容量
    BATCH_SIZE = 64      # バッチサイズ
    EPOCHS = 2           # 学習回数
    LR = 0.001           # 学習率

print("【GRU】データを読み込んでいます...")
train_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sub = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

train_df['comment_text'] = train_df['comment_text'].fillna("fillna")
test_df['comment_text'] = test_df['comment_text'].fillna("fillna")

print("前処理（URL削除など）を実行中...")

def clean_for_transformer(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # URL
    text = re.sub(r'<.*?>', '', text)                 # HTMLタグ
    text = re.sub(r'\n+', ' ', text)                  # 改行
    text = re.sub(r'\t+', ' ', text)                  # タブ
    text = re.sub(r' {2,}', ' ', text).strip()        # 連続スペース
    return text

train_df['comment_text'] = train_df['comment_text'].apply(clean_for_transformer)
test_df['comment_text'] = test_df['comment_text'].apply(clean_for_transformer)

def simple_clean(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower().split()

print("辞書を作成中...")
all_words = []
for text in train_df['comment_text'][:50000]:
    all_words.extend(simple_clean(text))

word_counts = Counter(all_words)
vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.most_common(Config.MAX_FEATURES))}
vocab_size = len(vocab) + 1 

def text_to_sequence(text, max_len):
    words = simple_clean(text)
    seq = [vocab.get(w, 0) for w in words] 
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

print("テキストをベクトル化中...")
X = np.array([text_to_sequence(t, Config.MAX_LEN) for t in train_df['comment_text']])
X_test = np.array([text_to_sequence(t, Config.MAX_LEN) for t in test_df['comment_text']])
y = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

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

train_loader = DataLoader(ToxicDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ToxicDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)
test_loader = DataLoader(ToxicDataset(X_test), batch_size=Config.BATCH_SIZE)

class BiGRU_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, Config.EMBED_DIM, padding_idx=0)
        self.gru = nn.GRU(Config.EMBED_DIM, Config.HIDDEN_DIM, num_layers=2, 
                          bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(Config.HIDDEN_DIM * 2, 6)
        
    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.gru(x)
        out, _ = torch.max(gru_out, dim=1)
        out = self.fc(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiGRU_Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
criterion = nn.BCEWithLogitsLoss()

print(f"学習を開始します (Device: {device})")

for epoch in range(Config.EPOCHS):
    model.train()
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1} / {Config.EPOCHS} 完了")

print("予測を実行中...")
model.eval()
preds = []

with torch.no_grad():
    for x_batch in test_loader:
        x_batch = x_batch.to(device)
        out = model(x_batch)
        preds.append(torch.sigmoid(out).cpu().numpy())

final_preds = np.concatenate(preds)
sub[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] = final_preds

sub.to_csv('submission_gru_cleaned.csv', index=False)
print("完了！ 'submission_gru_cleaned.csv' を保存しました。")
