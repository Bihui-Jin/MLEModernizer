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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
test = pd.read_csv(r"/kaggle/input/google-quest-challenge/test.csv")
target_cols = ['question_asker_intent_understanding', 'question_body_critical', 
               'question_conversational', 'question_expect_short_answer', 
               'question_fact_seeking', 'question_has_commonly_accepted_answer', 
               'question_interestingness_others', 'question_interestingness_self', 
               'question_multi_intent', 'question_not_really_a_question', 
               'question_opinion_seeking', 'question_type_choice',
               'question_type_compare', 'question_type_consequence',
               'question_type_definition', 'question_type_entity', 
               'question_type_instructions', 'question_type_procedure', 
               'question_type_reason_explanation', 'question_type_spelling', 
               'question_well_written', 'answer_helpful',
               'answer_level_of_information', 'answer_plausible', 
               'answer_relevance', 'answer_satisfaction', 
               'answer_type_instructions', 'answer_type_procedure', 
               'answer_type_reason_explanation', 'answer_well_written']

## === cell 2

train = pd.read_csv(r"/kaggle/input/google-quest-challenge/train.csv")
input_categories = list(train.columns[[1,2,5]]); input_categories

## === cell 3
from math import floor, ceil
from tqdm import tqdm
def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

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
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, max_sequence_length=512, t_max_len=30, q_max_len=239, a_max_len=239):
    
    question = "" if pd.isna(question) else question
    
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
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
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d"%(max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))
        
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
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
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
    
    test_preds = np.zeros((len(test), 30))
    
    model.eval();
    tk0 = tqdm(enumerate(test_loader))
    for idx, x_batch in tk0:
        with torch.no_grad():
            outputs = model(input_ids = x_batch[0].to(device), 
                            labels = None, 
                            attention_mask = x_batch[1].to(device),
                            token_type_ids = x_batch[2].to(device),
                           )
            predictions = outputs[0]
            test_preds[idx*batch_size : (idx+1)*batch_size] = predictions.detach().cpu().squeeze().numpy()

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()
    return output

## === cell 5
import torch
class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels = None):
        
        self.inputs = inputs
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.lengths = lengths

    def __getitem__(self, idx):
        
        input_ids       = self.inputs[0][idx]
        input_masks     = self.inputs[1][idx]
        input_segments  = self.inputs[2][idx]
        lengths         = self.lengths[idx]
        if self.labels is not None: # targets
            labels = self.labels[idx]
            return input_ids, input_masks, input_segments, labels, lengths
        return input_ids, input_masks, input_segments, lengths

    def __len__(self):
        return len(self.inputs[0])

## === cell 6
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification, BertConfig,
    WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import DataLoader
tokenizer = BertTokenizer.from_pretrained(r"/kaggle/input/bert-base-uncased/vocab.txt")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7

test_inputs = compute_input_arays(test, input_categories, tokenizer, max_sequence_length=512)
lengths_test = np.argmax(test_inputs[0] == 0, axis=1)
lengths_test[lengths_test == 0] = test_inputs[0].shape[1]
test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)
result = np.zeros((len(test), 30))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3687490740.py in <cell line: 0>()
----> 1 test_inputs = compute_input_arays(test, input_categories, tokenizer, max_sequence_length=512)
      2 lengths_test = np.argmax(test_inputs[0] == 0, axis=1)
      3 lengths_test[lengths_test == 0] = test_inputs[0].shape[1]
      4 test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
      5 test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)

NameError: name 'tokenizer' is not defined

## === cell 8
bert_model_config = f'/kaggle/input/bert-base-uncased/config.json'
bert_config = BertConfig.from_json_file(bert_model_config)
bert_config.num_labels = 30

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2739398666.py in <cell line: 0>()
      1 bert_model_config = f'/kaggle/input/bert-base-uncased/config.json'
----> 2 bert_config = BertConfig.from_json_file(bert_model_config)
      3 bert_config.num_labels = 30

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in from_json_file(cls, json_file)
    791 
    792         """
--> 793         config_dict = cls._dict_from_json_file(json_file)
    794         return cls(**config_dict)
    795 

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _dict_from_json_file(cls, json_file)
    796     @classmethod
    797     def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
--> 798         with open(json_file, encoding="utf-8") as reader:
    799             text = reader.read()
    800         return json.loads(text)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/bert-base-uncased/config.json'

## === cell 9
NUM_FOLDS = 5
result = np.zeros((len(test), 30))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BertForSequenceClassification.from_pretrained(f'/kaggle/input/bert-base-uncased', config=bert_config)
model = torch.nn.DataParallel(model)
model.to(device)
with torch.no_grad():
    for fold in range(1,NUM_FOLDS+1):
        model.load_state_dict(torch.load('/kaggle/input/pt-params/best_param_score_{}.pt'.format(fold)))

        fold_result = predict_result(model, test_loader)
        result += fold_result

result /= NUM_FOLDS

print(result)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/196054198.py in <cell line: 0>()
      5 # Initialize the BERT model with the correct number of output labels
      6 #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=bert_config)
----> 7 model = BertForSequenceClassification.from_pretrained(f'/kaggle/input/bert-base-uncased', config=bert_config)
      8 model = torch.nn.DataParallel(model)
      9 model.to(device)

NameError: name 'bert_config' is not defined

## === cell 10
submission = pd.read_csv(r'/kaggle/input/google-quest-challenge/sample_submission.csv')
raters = np.array([18, 18,  6,  6,  6,  6, 18, 18,  6,  6,  6,  6,  6,  6,  6,  6,  6, 6,  6,  3, 18, 18, 18, 18, 18, 90,  6,  6,  6, 18])

mins = np.min(result,axis=0)
maxs = np.max(result,axis=0)
result = (result - mins)/(maxs - mins)

result =  np.round(raters*result).astype(np.float)/raters
submission.loc[:, 'question_asker_intent_understanding':] = result
submission.to_csv('submission.csv', index=False)
submission.head()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2487614852.py in <cell line: 0>()
      6 result = (result - mins)/(maxs - mins)
      7 
----> 8 result =  np.round(raters*result).astype(np.float)/raters
      9 submission.loc[:, 'question_asker_intent_understanding':] = result
     10 submission.to_csv('submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/numpy/__init__.py in __getattr__(attr)
    322 
    323         if attr in __former_attrs__:
--> 324             raise AttributeError(__former_attrs__[attr])
    325 
    326         if attr == 'testing':

AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
