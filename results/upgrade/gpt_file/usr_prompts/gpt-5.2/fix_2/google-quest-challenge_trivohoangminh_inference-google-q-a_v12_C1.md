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

3.12

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

0.3404916644532028

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
test = pd.read_csv(r"/kaggle/input/google-quest-challenge/test.csv")

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



## === cell 2
train = pd.read_csv(r"/kaggle/input/google-quest-challenge/train.csv")
input_categories = list(train.columns[[1, 2, 5]])
input_categories



## === cell 3
from math import floor, ceil
from tqdm import tqdm
import torch


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")

    segments = []
    first_sep = True
    current_segment_id = 0

    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids


def _trim_input(
    title,
    question,
    answer,
    max_sequence_length=512,
    t_max_len=30,
    q_max_len=239,
    a_max_len=239,
):
    question = "" if pd.isna(question) else question

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len + q_len + a_len + 4) > max_sequence_length:
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
            q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len + a_new_len + q_new_len + 4 != max_sequence_length:
            raise ValueError(
                "New sequence length should be %d, but is %d"
                % (max_sequence_length, (t_new_len + a_new_len + q_new_len + 4))
            )

        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]

    return t, q, a


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows(), total=len(df)):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs(
            t, q, a, tokenizer, max_sequence_length
        )
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])




## === cell 4
def predict_result(model, test_loader, batch_size=64):
    test_preds = np.zeros((len(test_loader.dataset), 30), dtype=np.float32)

    model.eval()
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    for idx, x_batch in tk0:
        with torch.no_grad():
            outputs = model(
                input_ids=x_batch[0].to(device),
                labels=None,
                attention_mask=x_batch[1].to(device),
                token_type_ids=x_batch[2].to(device),
            )
            predictions = outputs[0]  # logits
            batch_np = predictions.detach().cpu().numpy()
            start = idx * batch_size
            end = start + batch_np.shape[0]
            test_preds[start:end] = batch_np

    output = torch.sigmoid(torch.from_numpy(test_preds)).numpy()
    return output




## === cell 5
class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels=None):
        self.inputs = inputs
        self.labels = labels if labels is not None else None
        self.lengths = lengths

    def __getitem__(self, idx):
        input_ids = self.inputs[0][idx]
        input_masks = self.inputs[1][idx]
        input_segments = self.inputs[2][idx]
        lengths = self.lengths[idx]
        if self.labels is not None:
            labels = self.labels[idx]
            return input_ids, input_masks, input_segments, labels, lengths
        return input_ids, input_masks, input_segments, lengths

    def __len__(self):
        return len(self.inputs[0])




## === cell 6
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader

BERT_DIR = "/kaggle/input/bert-base-uncased"

if not os.path.isdir(BERT_DIR):
    alt = "/kaggle/input/bert-base-uncased/bert-base-uncased"
    if os.path.isdir(alt):
        BERT_DIR = alt

tokenizer = BertTokenizer.from_pretrained(BERT_DIR, use_fast=False)

bert_config = BertConfig.from_pretrained(BERT_DIR)
bert_config.num_labels = 30



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
test_inputs = compute_input_arays(
    test, input_categories, tokenizer, max_sequence_length=512
)
lengths_test = np.argmax(test_inputs[0].numpy() == 0, axis=1)
lengths_test[lengths_test == 0] = test_inputs[0].shape[1]
test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
result = np.zeros((len(test), 30), dtype=np.float32)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/636453467.py in <cell line: 0>()
      1 test_inputs = compute_input_arays(
----> 2     test, input_categories, tokenizer, max_sequence_length=512
      3 )
      4 lengths_test = np.argmax(test_inputs[0].numpy() == 0, axis=1)
      5 lengths_test[lengths_test == 0] = test_inputs[0].shape[1]

NameError: name 'tokenizer' is not defined

## === cell 8
def _load_state_dict_flexible(model, state_dict):
    model_keys = list(model.state_dict().keys())
    sd_keys = list(state_dict.keys())
    if len(sd_keys) == 0:
        raise ValueError("Empty state_dict loaded.")

    model_has_module = model_keys[0].startswith("module.")
    sd_has_module = sd_keys[0].startswith("module.")

    if model_has_module and not sd_has_module:
        state_dict = {("module." + k): v for k, v in state_dict.items()}
    elif (not model_has_module) and sd_has_module:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)


NUM_FOLDS = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

model = BertForSequenceClassification.from_pretrained(BERT_DIR, config=bert_config)
model = torch.nn.DataParallel(model)
model.to(device)

result = np.zeros((len(test), 30), dtype=np.float32)

with torch.no_grad():
    for fold in range(1, NUM_FOLDS + 1):
        ckpt_path = f"/kaggle/input/pt-params/best_param_score_{fold}.pt"
        state = torch.load(ckpt_path, map_location="cpu")
        _load_state_dict_flexible(model, state)
        fold_result = predict_result(model, test_loader, batch_size=64)
        result += fold_result.astype(np.float32)

result /= NUM_FOLDS
print(result)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1508746372.py in <cell line: 0>()
     20 device = "cuda" if torch.cuda.is_available() else "cpu"
     21 
---> 22 model = BertForSequenceClassification.from_pretrained(BERT_DIR, config=bert_config)
     23 model = torch.nn.DataParallel(model)
     24 model.to(device)

NameError: name 'bert_config' is not defined

## === cell 9
submission = pd.read_csv(r"/kaggle/input/google-quest-challenge/sample_submission.csv")

raters = np.array(
    [
        18,
        18,
        6,
        6,
        6,
        6,
        18,
        18,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        3,
        18,
        18,
        18,
        18,
        18,
        90,
        6,
        6,
        6,
        18,
    ]
)

mins = np.min(result, axis=0)
maxs = np.max(result, axis=0)
den = maxs - mins
den[den == 0] = 1.0  # avoid divide-by-zero if any column is constant
result_norm = (result - mins) / den
result_norm = np.clip(result_norm, 0.0, 1.0)

result_rounded = (np.round(raters * result_norm).astype(float) / raters).astype(
    np.float32
)
result_rounded = np.clip(result_rounded, 0.0, 1.0)

submission.loc[:, target_cols] = result_rounded
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2675583129.py in <cell line: 0>()
     37 
     38 # Keep original post-processing logic, but fix numeric stability and np.float deprecation.
---> 39 mins = np.min(result, axis=0)
     40 maxs = np.max(result, axis=0)
     41 den = maxs - mins

NameError: name 'result' is not defined
