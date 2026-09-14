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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.13

# 3. Installed packages

gensim==4.4.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.1018038838623751

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.30334) has done: 'I map the original continuous scores to integer class indices (0‑4) so the classification loss receives correct targets, and I keep the reverse map for converting predictions back to the required score values. I also fix the soft‑max dimension during inference and use the proper class‑to‑score mapping, which should raise the Pearson correlation toward the target without altering the overall model architecture or training loops.'
- What this solution (achieved 0.19959) has done: 'I reduce the training length from ten epochs to a single epoch, which keeps the model architecture and training loop unchanged while intentionally lowering predictive performance. This should bring the Pearson correlation closer to the target score (≈0.10) without affecting any other part of the pipeline or the submission format.'
- What this solution (achieved 0.11718) has done: 'I keep the overall pipeline unchanged but make the neural network smaller and add stronger dropout (hidden size 128 instead of 1024 and dropout 0.9). This reduces model capacity and regularization, which is expected to lower the Pearson correlation from ~0.20 toward the target ≈ 0.10 while still producing a valid submission.csv.'
- What this solution (achieved 0.11823) has done: 'I slightly increase the dropout rate in the neural network (from 0.9 to 0.92). This keeps the same architecture and training loop but adds a touch more regularisation, which should modestly lower the Pearson correlation from 0.117 toward the target ≈ 0.102 without breaking the pipeline.'
- What this solution (achieved 0.02953) has done: 'I lower the model’s capacity a touch more by increasing the dropout rate from 0.92 to 0.95, which modestly reduces predictive performance and should move the Pearson correlation down toward the target 0.1018 without altering any other part of the pipeline. The rest of the code, data handling, training loop, and submission generation remain unchanged. The implementation below renumbers the cells starting at 1 as required.'

# 9. Code solution

## === cell 0
df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3512676240.py in <cell line: 0>()
----> 1 df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
      2 test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
      3 

NameError: name 'pd' is not defined

## === cell 1
from gensim.models import Word2Vec



## === cell 2
sentences = (
    [sentence.split() for sentence in df["target"]]
    + [sent.split() for sent in df["anchor"]]
    + [sentence.split() for sentence in test_df["target"]]
    + [sentence.split() for sentence in test_df["anchor"]]
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3407860960.py in <cell line: 0>()
      1 sentences = (
----> 2     [sentence.split() for sentence in df["target"]]
      3     + [sent.split() for sent in df["anchor"]]
      4     + [sentence.split() for sentence in test_df["target"]]
      5     + [sentence.split() for sentence in test_df["anchor"]]

NameError: name 'df' is not defined

## === cell 3
model = Word2Vec(
    sentences=sentences,
    vector_size=100,  # размер вектора
    window=5,  # окно контекста
    min_count=5,  # минимальная частота слова
    workers=1,  # количество потоков
    sg=1,  # 1 для skip-gram, 0 для CBOW
    epochs=10,  # количество эпох обучения
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3319241145.py in <cell line: 0>()
      1 model = Word2Vec(
----> 2     sentences=sentences,
      3     vector_size=100,  # размер вектора
      4     window=5,  # окно контекста
      5     min_count=5,  # минимальная частота слова

NameError: name 'sentences' is not defined

## === cell 4
def get_text_vector(text, model, vector_size=100):
    if not isinstance(text, str):
        return np.zeros(vector_size)

    words = text.split()
    vectors = []
    for word in words:
        try:
            vectors.append(model.wv[word])
        except KeyError:
            vectors.append(np.zeros(vector_size))

    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


score_to_idx = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}
idx_to_score = {v: k for k, v in score_to_idx.items()}

y = df["score"].map(score_to_idx).values.astype(int)

unique_values = pd.concat([df["context"], test_df["context"]]).unique().tolist()
encoding_dict = {v: i for i, v in enumerate(unique_values)}

df["context"] = df["context"].apply(lambda x: encoding_dict[x])

df["anchor_vector"] = df["anchor"].apply(lambda x: get_text_vector(x, model))
df["target_vector"] = df["target"].apply(lambda x: get_text_vector(x, model))

new_df = df.drop(columns=["anchor", "target", "id", "score"])

idies = test_df["id"]
test_df["context"] = test_df["context"].apply(lambda x: encoding_dict[x])
test_df["anchor_vector"] = test_df["anchor"].apply(lambda x: get_text_vector(x, model))
test_df["target_vector"] = test_df["target"].apply(lambda x: get_text_vector(x, model))

new_test_df = test_df.drop(columns=["anchor", "target", "id"])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/737334242.py in <cell line: 0>()
     19 idx_to_score = {v: k for k, v in score_to_idx.items()}
     20 
---> 21 y = df["score"].map(score_to_idx).values.astype(int)
     22 
     23 unique_values = pd.concat([df["context"], test_df["context"]]).unique().tolist()

NameError: name 'df' is not defined

## === cell 5
X_anchor = np.vstack(new_df["anchor_vector"].values)
X_target = np.vstack(new_df["target_vector"].values)
X_context = np.vstack(new_df["context"].values)
X = np.hstack([X_anchor, X_target, X_context])

X_test_anchor = np.vstack(new_test_df["anchor_vector"].values)
X_test_target = np.vstack(new_test_df["target_vector"].values)
X_test_context = np.vstack(new_test_df["context"].values)
X_test_df = np.hstack([X_test_anchor, X_test_target, X_test_context])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1228829179.py in <cell line: 0>()
----> 1 X_anchor = np.vstack(new_df["anchor_vector"].values)
      2 X_target = np.vstack(new_df["target_vector"].values)
      3 X_context = np.vstack(new_df["context"].values)
      4 X = np.hstack([X_anchor, X_target, X_context])
      5 

NameError: name 'np' is not defined

## === cell 6
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # smaller hidden layer
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(
            0.85
        )  # stronger regularisation than default but weaker than 0.95
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


input_size = X_train.shape[1]  # автоматическое определение размера входа
output_size = 5  # we have 5 classes (0‑4)
model = MyModel(input_size)

criterion = nn.CrossEntropyLoss() if output_size > 1 else nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3659355066.py in <cell line: 0>()
     18 
     19 X_train, X_test, y_train, y_test = train_test_split(
---> 20     X, y, test_size=0.2, random_state=42
     21 )
     22 

NameError: name 'X' is not defined

## === cell 7
from tqdm import tqdm



## === cell 8
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()

            if output_size > 1:  # классификация
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

    train_loss /= len(train_loader)
    test_loss /= len(test_loader)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    if output_size > 1:
        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    else:
        print(f"Test Loss: {test_loss:.4f} (Regression)")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3681448741.py in <cell line: 0>()
      3 
      4 for epoch in range(num_epochs):
----> 5     model.train()
      6     train_loss = 0.0
      7 

NameError: name 'model' is not defined

## === cell 9
with torch.no_grad():
    logits = model(torch.FloatTensor(X_test_df))
    probs = torch.softmax(logits, dim=1)  # softmax over classes
    pred_indices = torch.argmax(probs, dim=1).tolist()
answer = [idx_to_score[idx] for idx in pred_indices]



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2935458313.py in <cell line: 0>()
      1 with torch.no_grad():
----> 2     logits = model(torch.FloatTensor(X_test_df))
      3     probs = torch.softmax(logits, dim=1)  # softmax over classes
      4     pred_indices = torch.argmax(probs, dim=1).tolist()
      5 answer = [idx_to_score[idx] for idx in pred_indices]

NameError: name 'model' is not defined

## === cell 10
ans = pd.DataFrame({"id": idies, "score": answer})



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3943530828.py in <cell line: 0>()
----> 1 ans = pd.DataFrame({"id": idies, "score": answer})
      2 

NameError: name 'pd' is not defined

## === cell 11
ans.to_csv("submission.csv", index=False)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/899919599.py in <cell line: 0>()
----> 1 ans.to_csv("submission.csv", index=False)

NameError: name 'ans' is not defined
