# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
nltk==3.9.2
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
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)  # use TF‑IDF instead of CountVectorizer




## === cell 2
data_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"

data = pd.read_csv(data_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(test_path)
data.head(5)




## === cell 3
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = text.lower()  # lowercase text
    text = text.replace("\n", " ")  # remove new line
    text = text.replace(" ", " ")  # remove html space
    return text


data["full_text"] = data["full_text"].apply(clean_text)
test["full_text"] = test["full_text"].apply(clean_text)

data.head(5)




## === cell 4
stop_words = set(stopwords.words("english"))


def tokenize(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    processed_tokens = " ".join(tokens)
    return processed_tokens


data["full_text"] = data["full_text"].apply(tokenize)
test["full_text"] = test["full_text"].apply(tokenize)

data.head(5)




## === cell 5
import torch
import torch.nn as nn
import torch.optim as optim




## === cell 6
X = data["full_text"]
y = data["score"]

vectorizer = TfidfVectorizer()  # TF‑IDF feature extraction
X = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

X_train_tensor.shape, y_train_tensor.shape




## === cell 7
class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 256)  # larger hidden layer
        self.fc2 = nn.Linear(256, 128)  # larger hidden layer
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc4(x)
        return x




## === cell 8
model = NNModel()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




## === cell 9
epochs = 30  # train longer for better convergence
batch_size = 32

for epoch in range(epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i : i + batch_size]
        y_batch = y_train_tensor[i : i + batch_size]

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch + 1} Loss: {loss.item()}")




## === cell 10
with torch.no_grad():
    model.eval()
    y_pred = model(X_test_tensor)
    loss = criterion(y_pred, y_test_tensor)
    print(f"Validation Loss: {loss.item()}")




## === cell 11
X_test = test["full_text"]
X_test = vectorizer.transform(X_test)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)

with torch.no_grad():
    model.eval()
    y_pred = model(X_test_tensor)
    test["score"] = y_pred.numpy()
    rounded_scores = np.round(test["score"]).astype(int)
    rounded_scores = np.clip(rounded_scores, 1, 6)

    submission["score"] = rounded_scores

submission.head(5)




## === cell 12
submission = submission.drop("full_text", axis=1)
submission.to_csv("/kaggle/working/submission.csv", index=False)
