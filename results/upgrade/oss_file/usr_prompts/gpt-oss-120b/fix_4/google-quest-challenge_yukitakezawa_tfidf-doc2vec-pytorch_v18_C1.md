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

category_encoders==2.7.0
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scipy==1.15.3
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

0.18639

# 6. Current score

0.23425

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.23917) has done: 'I added a global definition of `target_cols` right after the imports so it’s available wherever needed, and updated `calc_score` to correctly reference the label matrix (using column indexing) and to divide by the number of target columns derived from the label shape rather than an undefined variable. These fixes resolve the NameError and ensure the Spearman‑based score is computed properly, allowing the script to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved 0.24112) has done: 'The fix resolves the shape mismatch when computing the Spearman score by extracting the true labels for the exact training/validation splits instead of using the full dataset, and it corrects the submission creation to use the test set’s `qa_id` rows rather than the sample file. These changes ensure the script runs end‑to‑end, produces a valid `submission.csv`, and retains the original modeling logic while keeping the score in the desired range.'
- What this solution (achieved 0.23425) has done: 'Implemented two key fixes:   
1️⃣ Updated `calc_score` to handle both (samples × targets) and (targets × samples) array shapes, preventing dimension mismatches during Spearman calculation.   
2️⃣ Corrected the training‑score block to keep predictions in the (samples × targets) layout and call the revised `calc_score`. These changes eliminate the runtime error and ensure a valid submission CSV is produced without altering the core model logic.'

# 9. Code solution

## === cell 0
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.one_hot import OneHotEncoder
import numpy as np
import torch
import torch.nn as nn

from torch.nn import Module
from torch.nn import Linear
from torch.nn import Sigmoid

from torch.nn import MSELoss
from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import gc

target_cols = [
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




## === cell 1
def load():
    train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
    test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")

    data_cols = ["question_title", "question_body", "answer", "category"]

    y_train = train[target_cols].copy()
    x_train = train[data_cols].copy()
    del train

    x_test = test.copy()
    del test

    text_encoder = Pipeline(
        [
            ("Text-TF-IDF", TfidfVectorizer(ngram_range=(1, 1))),
            ("Text-SVD", TruncatedSVD(n_components=100)),
        ],
        verbose=True,
    )

    ohe = OneHotEncoder(cols=["category"])

    preprocessor = ColumnTransformer(
        [
            ("Q-T", text_encoder, "question_title"),
            ("Q-B", text_encoder, "question_body"),
            ("A", text_encoder, "answer"),
            ("Category", ohe, "category"),
        ]
    )

    y_train = y_train.values

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    new_df = pd.concat([x_train, x_test])
    new_df = preprocessor.fit_transform(new_df)

    x_train = new_df[0:n_train]
    x_test = new_df[n_train:]

    return (
        x_train.astype(np.float32),
        y_train.astype(np.float32),
        x_test.astype(np.float32),
    )




## === cell 2
def calc_score(prediction, label):
    """
    Compute mean column‑wise Spearman correlation.
    Accepts either (samples, targets) or (targets, samples) shapes.
    """
    prediction = np.asarray(prediction)
    label = np.asarray(label)

    if prediction.shape[0] == label.shape[0] and prediction.shape[1] == label.shape[1]:
        pred_aligned = prediction
    else:
        pred_aligned = prediction.T

    num_cols = label.shape[1]  # number of target columns (30)
    score = 0.0
    for col_index in range(num_cols):
        corr, _ = spearmanr(pred_aligned[:, col_index], label[:, col_index])
        if np.isnan(corr):
            corr = 0.0
        score += corr / num_cols
    return score




## === cell 3
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_feature = 300
        self.linear1 = Linear(305, self.hidden_feature)
        self.sigmoid1 = Sigmoid()
        self.linear2 = Linear(self.hidden_feature, 30)
        self.sigmoid2 = Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        return x




## === cell 4
class MyDataset(Dataset):
    def __init__(self, data, label, device, transform=None):
        self.transform = transform
        self.data = torch.from_numpy(data).to(device)
        self.label = torch.from_numpy(label).to(device)
        self.data_num = self.data.shape[0]

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        return out_data, out_label


class TestDataset(Dataset):
    def __init__(self, data, device, transform=None):
        self.transform = transform
        self.data = torch.from_numpy(data).to(device)
        self.data_num = self.data.shape[0]

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        return out_data




## === cell 5
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

train_data, train_label, test_data = load()
dataset = MyDataset(train_data, train_label, device)
test_dataset = TestDataset(test_data, device)

train_size = int(len(dataset) * 1.0)  # using full data for training
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

model = Net().to(device)

epoch_size = 100
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=1e-5)

loss_list = np.zeros(epoch_size)

for epoch in range(epoch_size):
    for n_batch, (data, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        loss_list[epoch] += loss.item()
    loss_list[epoch] /= n_batch + 1
    print(f"{epoch}, loss is {loss_list[epoch]}")

plt.plot(loss_list)
plt.show()



## === cell 6
train_prediction = np.zeros((len(train_dataset), 30), dtype=np.float32)
train_true = np.zeros((len(train_dataset), 30), dtype=np.float32)

model.eval()
with torch.no_grad():
    for i, (input, label) in enumerate(train_dataset):
        output = model(input)
        train_prediction[i] = output.detach().cpu().numpy()
        train_true[i] = label.detach().cpu().numpy()

print(f"training score is {calc_score(train_prediction, train_true)}")



## === cell 7
if len(val_dataset) != 0:
    val_prediction = np.zeros((len(val_dataset), 30), dtype=np.float32)
    val_true = np.zeros((len(val_dataset), 30), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i, (input, label) in enumerate(val_dataset):
            output = model(input)
            val_prediction[i] = output.detach().cpu().numpy()
            val_true[i] = label.detach().cpu().numpy()

    val_prediction = val_prediction.T
    print(f"validation score is {calc_score(val_prediction, val_true)}")



## === cell 8
prediction = np.zeros((len(test_dataset), 30), dtype=np.float32)
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataset):
        output = model(data)
        prediction[i] = output.detach().cpu().numpy()



## === cell 9
test_df = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
submission = pd.DataFrame({"qa_id": test_df["qa_id"]})
prediction = prediction.T  # shape (30, n_test)

for i, col in enumerate(target_cols):
    submission[col] = prediction[i]

submission.to_csv("submission.csv", index=False)



## === cell 10
submission.head()



## === cell 11
df = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
df.describe()
