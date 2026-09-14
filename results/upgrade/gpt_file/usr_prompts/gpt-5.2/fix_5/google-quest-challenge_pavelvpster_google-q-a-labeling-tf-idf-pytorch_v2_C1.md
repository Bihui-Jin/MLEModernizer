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

0.26309

# 6. Current score

0.29607

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.32301) has done: 'Your current score (0.33517) is higher than the target (0.26309), so to move closer we should slightly reduce generalization performance with minimal, low-risk changes while preserving the same model/feature/training core logic. The smallest lever here is to reduce the latent dimensionality in the existing TruncatedSVD step (still TF‑IDF → SVD, same pipeline), which typically lowers Spearman a bit without breaking semantics. I also add deterministic seeds to stabilize the score around the new level (so it doesn’t jump back up/down unpredictably), and keep the submission formatting identical. No changes to the neural net architecture, loss, optimizer, epochs, or loops.'
- What this solution (achieved 0.31485) has done: 'Your current score (0.32301) is higher than the target (0.26309), so to move closer we should make a minimal change that slightly reduces model capacity while keeping the exact same TF‑IDF → SVD → MLP training core logic. The smallest reliable lever is reducing the SVD latent dimensionality a bit (this typically lowers rank-correlation performance without changing semantics). I only adjust `TruncatedSVD(n_components=40)` down to `n_components=25` and keep the deterministic seeding so the new score is stable and doesn’t bounce back upward. Everything else (features, model, loss, optimizer, epochs, submission writing) remains unchanged.'
- What this solution (achieved 0.30573) has done: 'Your current score (0.31485) is higher than the target (0.26309), so we should make a minimal, controlled reduction in model capacity to move the score downward toward the target band while keeping the exact same TF‑IDF → SVD → MLP core logic. The smallest reliable lever in your existing pipeline is the `TruncatedSVD(n_components=...)`; reducing it decreases representation richness and typically reduces Spearman without changing evaluation semantics. I only change `n_components` from 25 to 15 and keep everything else (seeds, model, loss, optimizer, epochs, submission writing) identical to preserve stability and runtime. This should nudge performance closer to 0.263 without risking invalid submissions.'
- What this solution (achieved 0.29607) has done: 'Your current score (0.30573) is still higher than the target (0.26309), so to move closer we should slightly reduce model capacity while keeping the exact same TF‑IDF → SVD → MLP training/prediction core logic intact. The smallest, most controlled lever in your existing pipeline is the `TruncatedSVD(n_components=...)`; reducing it typically lowers the mean Spearman without changing evaluation semantics or risking an invalid submission. I therefore lower `n_components` from 15 to 10 and keep everything else (seeding, TF‑IDF settings, MLP architecture, loss, optimizer, epochs, clipping, and submission format) unchanged for stability. This should nudge the score downward toward the target band with minimal risk.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv("../input/google-quest-challenge/train.csv", index_col="qa_id")
train.shape



## === cell 2
test = pd.read_csv("../input/google-quest-challenge/test.csv", index_col="qa_id")
test.shape



## === cell 3
train.head(3).T



## === cell 4
target_columns = [
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



## === cell 5
y_train = train[target_columns].copy()
x_train = train.drop(target_columns, axis=1)

x_test = test.copy()



## === cell 6
import random
import torch

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 7
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



## === cell 8
encoder = Pipeline(
    [
        ("TF-IDF", TfidfVectorizer(ngram_range=(1, 3))),
        ("SVD", TruncatedSVD(n_components=10, random_state=SEED)),
    ],
    verbose=True,
)



## === cell 9
preprocessor = ColumnTransformer(
    [
        ("Q-T", encoder, "question_title"),
        ("Q-B", encoder, "question_body"),
        ("A", encoder, "answer"),
    ],
    verbose=True,
)



## === cell 10
final_x_train = preprocessor.fit_transform(x_train)



## === cell 11
final_x_test = preprocessor.transform(x_test)



## === cell 12
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm



## === cell 13
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device




## === cell 14
class PyTorch:

    def fit(self, x_train, y_train):

        train_tensor = TensorDataset(
            torch.from_numpy(x_train.astype("float32")),
            torch.from_numpy(y_train.astype("float32")),
        )
        train_loader = DataLoader(train_tensor, batch_size=256, shuffle=True)

        self.model = nn.Sequential(
            weight_norm(nn.Linear(x_train.shape[1], 128)),
            nn.ReLU(),
            weight_norm(nn.Linear(128, 128)),
            nn.ReLU(),
            weight_norm(nn.Linear(128, y_train.shape[1])),
        ).to(device)

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight_v)
                nn.init.kaiming_normal_(m.weight_g)
                nn.init.constant_(m.bias, 0)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), betas=(0.9, 0.999), lr=1e-3
        )

        self.model.train()
        n_epochs = 50
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for train_part, y_part in train_loader:
                optimizer.zero_grad()
                y_pred_part = self.model(train_part.to(device))
                loss = criterion(y_pred_part, y_part.to(device))
                loss.backward()
                optimizer.step()
                epoch_loss += y_pred_part.shape[0] * loss.item()
            print(
                "Epoch %3d / %3d. Loss = %.5f"
                % (epoch + 1, n_epochs, epoch_loss / x_train.shape[0])
            )

    def predict(self, x):
        self.model.eval()
        tensor = torch.from_numpy(x.astype("float32"))
        loader = DataLoader(tensor, batch_size=256, shuffle=False)
        y_pred = np.empty((0, len(target_columns)))
        with torch.no_grad():
            for x_part in loader:
                y_pred_part = self.model(x_part.to(device)).data.cpu().numpy()
                y_pred = np.append(y_pred, y_pred_part, axis=0)
        return y_pred




## === cell 15
pytorch = PyTorch()
pytorch.fit(final_x_train, y_train.values)
y_pred = pytorch.predict(final_x_test)



## === cell 16
submission = pd.read_csv(
    "../input/google-quest-challenge/sample_submission.csv", index_col="qa_id"
)
submission.shape



## === cell 17
for idx, column in enumerate(target_columns):
    submission[column] = np.clip(y_pred[:, idx], 0.0, 1.0)



## === cell 18
submission.head()



## === cell 19
submission.to_csv("submission.csv")
