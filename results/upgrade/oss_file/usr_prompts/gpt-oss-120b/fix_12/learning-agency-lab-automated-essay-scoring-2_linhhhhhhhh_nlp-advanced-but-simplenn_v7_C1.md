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

# 5. Target score

0.72383

# 6. Current score

0.51576

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.56746) has done: 'The changes speed up data loading by batching sparse rows into a single dense matrix instead of converting each row individually, increase the batch size to reduce loop overhead, and remove unnecessary worker processes. Small seed and backend settings ensure deterministic behavior without affecting model logic. These tweaks keep the exact model architecture, training loop, and evaluation unchanged while making the whole pipeline finish well under the 600‑second limit.'
- What this solution (achieved 0.0) has done: 'I add class‑balanced weighting to the cross‑entropy loss, introduce a modest learning‑rate scheduler and double the training epochs (to 60) so the model can learn more without changing its architecture. These tweaks are small, keep the core pipeline intact, and are expected to raise the quadratic weighted kappa toward the target 0.72383.'
- What this solution (achieved 0.48424) has done: 'The changes replace the dense‑conversion collate functions with true sparse tensors and adjust the model’s first linear layer to operate directly on sparse inputs, eliminating the expensive per‑batch dense conversion. The DataLoader now returns a `torch.sparse_coo_tensor`, and the forward pass uses `torch.sparse.mm` for the first layer while keeping the remaining layers dense. The number of workers is set to 0 to avoid extra inter‑process overhead. These modifications keep the exact architecture, loss, optimizer, and training schedule, preserving result accuracy while drastically reducing runtime.'
- What this solution (achieved 0.51352) has done: 'I add a small, targeted change in the prediction step: instead of using the arg‑max of the logits, I compute the soft‑max probabilities and take the expected score (a weighted average of class indices). This often yields more calibrated predictions for quadratic weighted kappa and should push the score upward toward the target while keeping the model architecture unchanged. I also import the needed functional module.'
- What this solution (achieved 0.51576) has done: 'I keep the overall pipeline unchanged but tweak the training hyper‑parameters slightly to let the model learn more effectively. The optimizer is switched to AdamW with a small weight‑decay, the learning‑rate scheduler now steps every 15 epochs (instead of 20) and the training loop runs for 80 epochs rather than 60. These minimal adjustments are expected to improve the quadratic weighted kappa score, moving it closer to the target while preserving the original architecture and data handling.'

# 9. Code solution

## === cell 0
import string
import nltk
from nltk.corpus import stopwords
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse  # added for efficient batch conversion
import random



## === cell 1
data = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
data1 = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
data2 = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)



## === cell 2
nltk.download("stopwords", quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def preprocess_text(self, text):
        text = text.replace("\n", " ")
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        processed_text = " ".join(filtered_tokens)
        return processed_text

    def preprocess_dataframe(self, df):
        df.drop(columns=["essay_id"], inplace=True)
        df["full_text"] = df["full_text"].apply(self.preprocess_text)
        return df




## === cell 3
preprocessor = TextPreprocessor()
df = preprocessor.preprocess_dataframe(data)



## === cell 4
df



## === cell 5
x = df["full_text"]
y = df["score"]



## === cell 6
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, dtype=np.float32)
X = vectorizer.fit_transform(x)



## === cell 7
X_train_sp, X_test_sp, y_train_arr, y_test_arr = train_test_split(
    X, y.values, test_size=0.2, random_state=42
)



## === cell 8
y_train_vec = torch.tensor(y_train_arr - 1, dtype=torch.long)
y_test_vec = torch.tensor(y_test_arr - 1, dtype=torch.long)

class_counts = np.bincount(y_train_arr - 1, minlength=6)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights * (len(class_weights) / class_weights.sum())
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 9
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1_weight = nn.Parameter(torch.randn(input_size, 64))
        self.fc1_bias = nn.Parameter(torch.zeros(64))

        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 6)

    def forward(self, x):
        if x.is_sparse:
            x = torch.sparse.mm(x, self.fc1_weight) + self.fc1_bias
        else:
            x = torch.addmm(self.fc1_bias, x, self.fc1_weight)
        x = torch.relu(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # logits
        return x


input_size = X_train_sp.shape[1]
model = SimpleNN(input_size).to(device)  # move model to device

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)




## === cell 10
class SparseDataset(Dataset):
    def __init__(self, X_sp, y_tensor):
        self.X_sp = X_sp  # scipy CSR matrix
        self.y = y_tensor  # torch long tensor

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X_sp[idx], self.y[idx]


def csr_batch_to_sparse_tensor(batch_csr):
    batch_coo = batch_csr.tocoo()
    indices = torch.tensor([batch_coo.row, batch_coo.col], dtype=torch.long)
    values = torch.tensor(batch_coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=batch_coo.shape).coalesce()


def sparse_collate_fn(batch):
    rows, targets = zip(*batch)  # list of csr rows, list of tensors
    batch_csr = scipy.sparse.vstack(rows)
    inputs = csr_batch_to_sparse_tensor(batch_csr)
    targets = torch.stack(list(targets))
    return inputs, targets


batch_size = 1024
train_dataset = SparseDataset(X_train_sp, y_train_vec)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=sparse_collate_fn,
    num_workers=0,  # avoid extra multiprocessing overhead
    pin_memory=True,
)

num_epochs = 80
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss = loss.item()
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")



## === cell 11
test_dataset = SparseDataset(X_test_sp, y_test_vec)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=sparse_collate_fn,
    num_workers=0,
    pin_memory=True,
)

model.eval()
test_loss_total = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss_total += loss.item()
print(f"Test Loss: {test_loss_total / len(test_loader):.4f}")



## === cell 12
df1 = preprocessor.preprocess_dataframe(data1)



## === cell 13
X_new_sp = vectorizer.transform(df1["full_text"])




## === cell 14
class SparsePredictDataset(Dataset):
    def __init__(self, X_sp):
        self.X_sp = X_sp

    def __len__(self):
        return self.X_sp.shape[0]

    def __getitem__(self, idx):
        return self.X_sp[idx]


def predict_collate_fn(batch):
    batch_csr = scipy.sparse.vstack(batch)
    return csr_batch_to_sparse_tensor(batch_csr)


predict_loader = DataLoader(
    SparsePredictDataset(X_new_sp),
    batch_size=batch_size,
    collate_fn=predict_collate_fn,
    num_workers=0,
    pin_memory=True,
)

model.eval()
all_preds = []
with torch.no_grad():
    for batch in predict_loader:
        batch = batch.to(device, non_blocking=True)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        class_range = torch.arange(
            1, 7, device=device, dtype=torch.float32
        )  # scores 1..6
        expected_scores = torch.sum(probs * class_range, dim=1)
        preds = torch.round(expected_scores).cpu().numpy().astype(int)
        all_preds.append(preds)

pred_classes = np.concatenate(all_preds)
predicted_scores = np.clip(pred_classes, 1, 6)
df1["score"] = predicted_scores.astype(int)



## === cell 15
df_submission = pd.concat([data2["essay_id"], df1["score"]], axis=1)



## === cell 16
df_submission.to_csv("submission.csv", index=False)
