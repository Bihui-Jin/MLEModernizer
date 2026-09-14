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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.7034798860549927

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

is_inference_flag = True
try:
    tweet_models_dir = os.listdir("/kaggle/input/albert-xl-300/")
    if len(tweet_models_dir) > 0:
        is_inference_flag = True
except Exception:
    is_inference_flag = False



## === cell 1
print("Inference flag status :", is_inference_flag)



## === cell 2
if not is_inference_flag:
    raise RuntimeError(
        "Inference model directory /kaggle/input/albert-xl-300/ not found. "
        "This script is configured to run inference-only in this environment."
    )



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3396377797.py in <cell line: 0>()
      3 # This competition solution expects pretrained weights in /kaggle/input/albert-xl-300/ for inference.
      4 if not is_inference_flag:
----> 5     raise RuntimeError(
      6         "Inference model directory /kaggle/input/albert-xl-300/ not found. "
      7         "This script is configured to run inference-only in this environment."

RuntimeError: Inference model directory /kaggle/input/albert-xl-300/ not found. This script is configured to run inference-only in this environment.

## === cell 3
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split



## === cell 4
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sub_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")




## === cell 5
def jaccard(str1, str2):
    a = str1.lower().split()
    b = str2.lower().split()
    c = set(a).intersection(set(b))
    denom = len(a) + len(b) - len(c)
    if denom == 0:
        return 0.0
    return float(len(c)) / denom




## === cell 6
train_df.dropna(inplace=True)



## === cell 7
X_train, X_test = train_test_split(train_df, test_size=0.10, random_state=42)



## === cell 8
train = np.array(X_train)
val = np.array(X_test)
test = np.array(test_df)
use_cuda = True



## === cell 9
os.makedirs("data/models/albert", exist_ok=True)
os.makedirs("data/models/roberta", exist_ok=True)



## === cell 10
"""
Prepare training data in QA-compatible format (kept for completeness).
In this environment we run inference-only, so we do not write train/val json.
"""


def find_all(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def do_qa_train(train_arr):
    output = {}
    output["version"] = "v1.0"
    output["data"] = []
    paragraphs = []
    for line in train_arr:
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(answer) != str or type(context) != str or type(question) != str:
            continue
        answer_starts = find_all(context, answer)
        for answer_start in answer_starts:
            answers.append({"answer_start": answer_start, "text": answer.lower()})
            break
        qas.append(
            {
                "question": question,
                "id": qid,
                "is_impossible": False,
                "answers": answers,
            }
        )
        paragraphs.append({"context": context.lower(), "qas": qas})

    output["data"].append({"title": "None", "paragraphs": paragraphs})
    return output




## === cell 11
"""
Prepare testing data in QA-compatible format
"""


def convert_test_qa_json(test_arr):
    output = {}
    output["version"] = "v1.0"
    output["data"] = []
    paragraphs = []
    for line in test_arr:
        context = line[1]
        qas = []
        question = line[-1]
        qid = line[0]
        if type(context) != str or type(question) != str:
            continue
        answers = []
        answers.append({"answer_start": 1000000, "text": "__None__"})
        qas.append(
            {
                "question": question,
                "id": qid,
                "is_impossible": False,
                "answers": answers,
            }
        )
        paragraphs.append({"context": context.lower(), "qas": qas})

    output["data"].append({"title": "None", "paragraphs": paragraphs})
    return output


qa_test = convert_test_qa_json(test)
with open("data/test.json", "w") as outfile:
    json.dump(qa_test, outfile)



## === cell 12
pass



## === cell 13
pass



## === cell 14
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")



## === cell 15
import torch
from torch.utils.data import DataLoader, SequentialSampler
from collections import OrderedDict

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import SquadResult, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits

if not is_inference_flag:
    model_name_or_path = "data/models/"
else:
    model_name_or_path = "/kaggle/input/albert-xl-300/"

output_dir = ""

n_best_size = 1
max_answer_length = 254
do_lower_case = True
null_score_diff_threshold = 0.0


def to_list(tensor):
    return tensor.detach().cpu().tolist()


config_class, model_class, tokenizer_class = (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
model = model_class.from_pretrained(model_name_or_path, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def run_prediction(question_texts, context_text):
    """Setup function to compute predictions"""
    if question_texts[0] != "neutral":
        examples = []
        for i, question_text in enumerate(question_texts):
            example = SquadExample(
                qas_id=str(i),
                question_text=question_text,
                context_text=context_text,
                answer_text=None,
                start_position_character=None,
                title="Predict",
                is_impossible=False,
                answers=None,
            )
            examples.append(example)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=300,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            return_dataset="pt",
            threads=1,
        )

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

        all_results = []
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                example_indices = batch[3]
                outputs = model(**inputs)

                for j, example_index in enumerate(example_indices):
                    eval_feature = features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    output = [to_list(o[j]) for o in outputs]
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)
                    all_results.append(result)

        output_prediction_file = "predictions.json"
        output_nbest_file = "nbest_predictions.json"
        output_null_log_odds_file = "null_predictions.json"

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            n_best_size,
            max_answer_length,
            do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            False,  # verbose_logging
            True,  # version_2_with_negative
            null_score_diff_threshold,
            tokenizer,
        )
    else:
        predictions = OrderedDict([(0, context_text)])

    return predictions




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 16
jaccard_scores = []
predictions_x_test = []

for _, row in X_test.head(10).iterrows():
    context = row["text"]
    selected_text = row["selected_text"]
    questions = [row["sentiment"]]
    preds_dict = run_prediction(questions, context)
    for key in preds_dict.keys():
        predicted_text = preds_dict[key]
        predictions_x_test.append(
            {
                "selected_text": selected_text,
                "predicted_text": predicted_text,
                "sentiment": row["sentiment"],
                "textID": row["textID"],
            }
        )
        jaccard_scores.append(jaccard(selected_text, predicted_text))

print(
    "Jaccard Score (sample of 10):",
    float(np.mean(jaccard_scores)) if len(jaccard_scores) else 0.0,
)
predictions_x_test_df = pd.DataFrame.from_dict(predictions_x_test)
print(predictions_x_test_df.head())



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1582451147.py in <cell line: 0>()
      7     selected_text = row["selected_text"]
      8     questions = [row["sentiment"]]
----> 9     preds_dict = run_prediction(questions, context)
     10     for key in preds_dict.keys():
     11         predicted_text = preds_dict[key]

NameError: name 'run_prediction' is not defined

## === cell 17
print("test_df shape:", test_df.shape)



## === cell 18
predictions = []

for _, row in test_df.iterrows():
    context = row["text"]
    questions = [row["sentiment"]]
    preds_dict = run_prediction(questions, context)
    for key in preds_dict.keys():
        predicted_text = preds_dict[key]
        predictions.append({"textID": row["textID"], "selected_text": predicted_text})

predictions_df = pd.DataFrame.from_dict(predictions)

predictions_df = predictions_df.drop_duplicates(subset=["textID"], keep="first")
output_df = sub_df[["textID"]].merge(predictions_df, on="textID", how="left")

output_df["selected_text"] = output_df["selected_text"].fillna("")

output_path = "submission.csv"
output_df.to_csv(output_path, index=False)

print(output_df.head())
print("Wrote:", output_path, "rows:", len(output_df), "cols:", list(output_df.columns))

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/482283604.py in <cell line: 0>()
      5     context = row["text"]
      6     questions = [row["sentiment"]]
----> 7     preds_dict = run_prediction(questions, context)
      8     # There is only one qas_id="0" per row; still keep original loop semantics.
      9     for key in preds_dict.keys():

NameError: name 'run_prediction' is not defined
