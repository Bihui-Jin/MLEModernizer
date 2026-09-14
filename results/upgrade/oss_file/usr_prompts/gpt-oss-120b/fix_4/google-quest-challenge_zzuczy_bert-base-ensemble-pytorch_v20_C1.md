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

0.3655933516945904

# 6. Current score

0.23157

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03533) has done: 'I fix the import errors, make tokenisation fallback to the standard HuggingFace model, guard directory creation, skip saving/loading non‑existent pre‑processed files, and build a simple dataset that tokenises the test rows on the fly. This lets the script run end‑to‑end, load the available pretrained model weights, generate predictions for the test set, and write a correctly formatted `submission.csv` file.'
- What this solution (achieved 0.16444) has done: 'I replace the failing Transformer imports with safe fallbacks, add a simple deterministic tokeniser for cases without a tokenizer, and compute predictions using a lightweight linear‑regression model built from question/answer text lengths. This removes the crash‑prone BERT model while still providing varied predictions that correlate better with the targets, moving the score toward the target range. The script now writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.23157) has done: 'I added the missing `os` import to fix the directory‑creation errors, and expanded the feature engineering in `build_features` to include word counts and simple punctuation flags for title, body and answer texts. These extra numeric signals give the linear regression a richer representation, which should raise the Spearman correlation toward the target while keeping the core model unchanged. The rest of the pipeline remains the same, now successfully creating the required `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import pandas as pd
import numpy as np

BertTokenizer = None
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
if BertTokenizer is not None:
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        tokenizer = None
else:
    tokenizer = None



## === cell 7
max_length_map = {"question_title": 32, "question_body": 512, "answer": 512}




## === cell 8
def txt_re(txt):
    txt = txt.strip()
    txt = re.sub("https?.*$", "", txt)
    txt = re.sub("https?.*\s", "", txt)
    txt = re.sub("\n+", " ", txt)
    txt = re.sub("\r+", " ", txt)
    txt = re.sub("\t+", " ", txt)
    txt = re.sub("&gt;", ">", txt)
    txt = re.sub("&lt;", "<", txt)
    txt = re.sub("&amp;", "&", txt)
    txt = re.sub("&quot;", '"', txt)
    return txt




## === cell 9
VOCAB_SIZE = 30522  # BERT's vocab size (used for simple fallback tokenisation)


def simple_tokenize(text, max_length):
    """
    Very light‑weight tokeniser used when HuggingFace tokenizer is unavailable.
    It splits on whitespace, hashes each token and maps it into the token id range.
    """
    tokens = text.split()
    ids = [abs(hash(tok)) % VOCAB_SIZE for tok in tokens][:max_length]
    padding = [0] * (max_length - len(ids))
    return ids + padding


def get_input(txt, pair_txt, tokenizer, max_length):
    txt = txt_re(txt)
    if tokenizer is not None:
        encoded = tokenizer.encode_plus(
            txt,
            pair_txt,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].flatten().astype(np.int64).tolist()
        segment_masks = encoded["token_type_ids"].flatten().astype(np.int64).tolist()
        input_masks = encoded["attention_mask"].flatten().astype(np.int64).tolist()
    else:
        input_ids = simple_tokenize(txt, max_length)
        segment_masks = [0] * max_length
        input_masks = [0] * max_length
    return input_ids, segment_masks, input_masks




## === cell 10
def computer_input_array(df):
    q_input_ids, q_segment_masks, q_input_masks = [], [], []
    a_input_ids, a_segment_masks, a_input_masks = [], [], []
    for _, instance in df[input_columns].iterrows():
        title, question, answer = (
            instance.question_title,
            instance.question_body,
            instance.answer,
        )

        input_ids, segment_masks, input_masks = get_input(
            title, question, tokenizer, max_length_map["question_body"]
        )
        q_input_ids.append(input_ids)
        q_segment_masks.append(segment_masks)
        q_input_masks.append(input_masks)

        input_ids, segment_masks, input_masks = get_input(
            answer, None, tokenizer, max_length_map["answer"]
        )
        a_input_ids.append(input_ids)
        a_segment_masks.append(segment_masks)
        a_input_masks.append(input_masks)
    question = [
        [input_id, segment_mask, input_mask]
        for input_id, segment_mask, input_mask in zip(
            q_input_ids, q_segment_masks, q_input_masks
        )
    ]
    answer = [
        [input_id, segment_mask, input_mask]
        for input_id, segment_mask, input_mask in zip(
            a_input_ids, a_segment_masks, a_input_masks
        )
    ]

    return question, answer




## === cell 11
question_train, answer_train = computer_input_array(train)
question_test, answer_test = computer_input_array(test)



## === cell 12
answer_test  # just to confirm execution; no error now



## === cell 13
labels = train[output_columns].values.tolist()




## === cell 14
def build_features(df):
    title_len = df.question_title.astype(str).apply(len).values
    body_len = df.question_body.astype(str).apply(len).values
    answer_len = df.answer.astype(str).apply(len).values

    title_wc = df.question_title.astype(str).apply(lambda x: len(x.split())).values
    body_wc = df.question_body.astype(str).apply(lambda x: len(x.split())).values
    answer_wc = df.answer.astype(str).apply(lambda x: len(x.split())).values

    title_q = df.question_title.astype(str).apply(lambda x: int("?" in x)).values
    body_q = df.question_body.astype(str).apply(lambda x: int("?" in x)).values
    answer_q = df.answer.astype(str).apply(lambda x: int("?" in x)).values

    bias = np.ones_like(title_len)

    feats = np.vstack(
        [
            title_len,
            body_len,
            answer_len,
            title_wc,
            body_wc,
            answer_wc,
            title_q,
            body_q,
            answer_q,
            bias,
        ]
    ).T
    return feats


X_train = build_features(train)  # (n_train, 10)
Y_train = train[output_columns].values  # (n_train, 30)

coeffs, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond=None)

X_test = build_features(test)  # (n_test, 10)
linear_preds = X_test @ coeffs  # (n_test, 30)

avg_preds = np.clip(linear_preds, 0.0, 1.0)



## === cell 15
os.makedirs("./data", exist_ok=True)
os.makedirs("./model", exist_ok=True)



## === cell 16
import torch

torch.save(
    {"question": question_train, "answer": answer_train, "label": labels},
    "./data/train_data.t7",
)
torch.save({"question": question_test, "answer": answer_test}, "./data/test_data.t7")



## === cell 17
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

try:
    from transformers import AdamW
except Exception:
    from torch.optim import Adam as AdamW




## === cell 18
class Model_v1(nn.Module):
    def __init__(self):
        super().__init__()
        if BertModel is None:
            raise RuntimeError(
                "BertModel not available; Model_v1 cannot be instantiated."
            )
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # replace fixed 2‑D pool with 1‑D avg

        self.output = nn.Linear(768 * 2, 30)

    def forward(
        self,
        q_inputs,
        q_input_masks,
        q_segment_masks,
        a_inputs,
        a_input_masks,
        a_segment_masks,
    ):

        q_outputs = self.bert(
            input_ids=q_inputs,
            attention_mask=q_input_masks,
            token_type_ids=q_segment_masks,
        )
        q_x = q_outputs.last_hidden_state  # (b, seq, 768)
        q_x = self.dropout(q_x)
        q_x = q_x.permute(0, 2, 1)  # (b, 768, seq)
        q_x = self.pool(q_x).squeeze(-1)  # (b, 768)

        a_outputs = self.bert(
            input_ids=a_inputs,
            attention_mask=a_input_masks,
            token_type_ids=a_segment_masks,
        )
        a_x = a_outputs.last_hidden_state
        a_x = self.dropout(a_x)
        a_x = a_x.permute(0, 2, 1)
        a_x = self.pool(a_x).squeeze(-1)

        t_q_a = torch.cat((q_x, a_x), dim=1)
        output = self.output(t_q_a)
        return torch.sigmoid(output)




## === cell 19
models = []
for fold in range(5):
    model_path = f"../input/google-qa-labeling-pretrained-v3/model_{fold}_2.t7"
    if os.path.exists(model_path):
        try:
            model = Model_v1()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            models.append(model)
            print(f"Loaded model for fold {fold}")
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")

if len(models) == 0:
    print(
        "No pretrained models found – using linear regression predictions as placeholder."
    )




## === cell 20
class QA_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len_map):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len_map = max_len_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        q_ids, q_seg, q_att = get_input(
            row.question_title,
            row.question_body,
            self.tokenizer,
            self.max_len_map["question_body"],
        )
        a_ids, a_seg, a_att = get_input(
            row.answer, None, self.tokenizer, self.max_len_map["answer"]
        )
        return (
            torch.tensor(q_ids, dtype=torch.long),
            torch.tensor(q_att, dtype=torch.long),
            torch.tensor(q_seg, dtype=torch.long),
            torch.tensor(a_ids, dtype=torch.long),
            torch.tensor(a_att, dtype=torch.long),
            torch.tensor(a_seg, dtype=torch.long),
        )


test_dataset = QA_Dataset(test, tokenizer, max_length_map)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)



## === cell 21
final_predicts = []
for model in models:
    preds_per_row = []
    with torch.no_grad():
        for q_ids, q_att, q_seg, a_ids, a_att, a_seg in test_loader:
            scores = model(q_ids, q_att, q_seg, a_ids, a_att, a_seg)
            preds_per_row.append(scores.cpu().numpy())
    preds_per_row = np.concatenate(preds_per_row, axis=0)
    final_predicts.append(preds_per_row)



## === cell 22
if final_predicts:
    avg_preds = np.mean(np.stack(final_predicts, axis=0), axis=0)  # (num_samples, 30)



## === cell 23
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



## === cell 24
submission = pd.DataFrame(avg_preds, columns=output_cols)
submission.insert(0, "qa_id", test["qa_id"].values)
submission = submission[sample_submission.columns]
submission.head()



## === cell 25
submission.to_csv("submission.csv", index=False)
print("submission.csv written, shape:", submission.shape)
