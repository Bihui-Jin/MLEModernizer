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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.12

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
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.09453

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/train.csv')
test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
sample_submission = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/sample_submission.csv')

vectorizer = TfidfVectorizer(max_features=1000)
train_tfidf = vectorizer.fit_transform(train['response_a'] + " " + train['response_b'])
test_tfidf = vectorizer.transform(test['response_a'] + " " + test['response_b'])

train['response_a_length'] = train['response_a'].apply(len)
train['response_b_length'] = train['response_b'].apply(len)
train['response_a_word_count'] = train['response_a'].apply(lambda x: len(x.split()))
train['response_b_word_count'] = train['response_b'].apply(lambda x: len(x.split()))

test['response_a_length'] = test['response_a'].apply(len)
test['response_b_length'] = test['response_b'].apply(len)
test['response_a_word_count'] = test['response_a'].apply(lambda x: len(x.split()))
test['response_b_word_count'] = test['response_b'].apply(lambda x: len(x.split()))

X_train_combined = np.hstack((train_tfidf.toarray(), train[['response_a_length', 'response_b_length', 'response_a_word_count', 'response_b_word_count']].values))
X_test_combined = np.hstack((test_tfidf.toarray(), test[['response_a_length', 'response_b_length', 'response_a_word_count', 'response_b_word_count']].values))

y_a = train['winner_model_a'].values
y_b = train['winner_model_b'].values
y_tie = train['winner_tie'].values

class ChatbotDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train_a, X_val_a, y_train_a, y_val_a = train_test_split(X_train_combined, y_a, test_size=0.2, random_state=42)
X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(X_train_combined, y_b, test_size=0.2, random_state=42)
X_train_tie, X_val_tie, y_train_tie, y_val_tie = train_test_split(X_train_combined, y_tie, test_size=0.2, random_state=42)

train_dataset_a = ChatbotDataset(X_train_a, y_train_a)
val_dataset_a = ChatbotDataset(X_val_a, y_val_a)
train_dataset_b = ChatbotDataset(X_train_b, y_train_b)
val_dataset_b = ChatbotDataset(X_val_b, y_val_b)
train_dataset_tie = ChatbotDataset(X_train_tie, y_train_tie)
val_dataset_tie = ChatbotDataset(X_val_tie, y_val_tie)

train_loader_a = DataLoader(train_dataset_a, batch_size=32, shuffle=True)
val_loader_a = DataLoader(val_dataset_a, batch_size=32, shuffle=False)
train_loader_b = DataLoader(train_dataset_b, batch_size=32, shuffle=True)
val_loader_b = DataLoader(val_dataset_b, batch_size=32, shuffle=False)
train_loader_tie = DataLoader(train_dataset_tie, batch_size=32, shuffle=True)
val_loader_tie = DataLoader(val_dataset_tie, batch_size=32, shuffle=False)

class ChatbotModel(nn.Module):
    def __init__(self, input_dim):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

input_dim = X_train_combined.shape[1]
model_a = ChatbotModel(input_dim).to(device)
model_b = ChatbotModel(input_dim).to(device)
model_tie = ChatbotModel(input_dim).to(device)

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).view(-1, 1)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

train_model(model_a, train_loader_a, val_loader_a)
train_model(model_b, train_loader_b, val_loader_b)
train_model(model_tie, train_loader_tie, val_loader_tie)

test_tensor = torch.tensor(X_test_combined, dtype=torch.float32).to(device)
model_a.eval()
model_b.eval()
model_tie.eval()

with torch.no_grad():
    test_predictions_a = model_a(test_tensor).cpu().numpy()
    test_predictions_b = model_b(test_tensor).cpu().numpy()
    test_predictions_tie = model_tie(test_tensor).cpu().numpy()

submission = pd.DataFrame({
    'id': test['id'],
    'winner_model_a': test_predictions_a.flatten(),
    'winner_model_b': test_predictions_b.flatten(),
    'winner_tie': test_predictions_tie.flatten()
})

submission.to_csv('submission.csv', index=False)
