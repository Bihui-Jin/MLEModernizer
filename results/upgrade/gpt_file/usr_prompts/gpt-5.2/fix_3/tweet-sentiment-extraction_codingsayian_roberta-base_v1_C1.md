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

3.10

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
transformers==4.53.3
vega-datasets==0.9.0

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

0.7191751003265381

# 6. Current score

0.30638

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.29108) has done: 'The crashes come from trying to load a tokenizer/model from non-existent local Kaggle input folders (`../input/robertatokenizer/` and `../input/robertamodel`), which leaves `tokenizer`/`model` undefined and breaks all downstream cells. I minimally fix this by (1) using the standard `roberta-base` tokenizer/model from Hugging Face cache (offline-safe on Kaggle) with a local-only fallback, and (2) ensuring TensorFlow is imported (needed for `to_tf_dataset`/`model.predict`). I also fix a small logic bug in `predict_answers` where `answer` could be referenced before assignment if no valid span is found, by falling back to the full tweet text. These changes preserve the original QA-span extraction approach and produce a valid `submission.csv`.'
- What this solution (achieved 0.30638) has done: 'I fix the runtime error caused by an incompatible `protobuf`/`transformers` import path in this Kaggle environment by pinning protobuf to the pure-Python implementation via an environment variable before importing `transformers` (this resolves the `MessageFactory.GetPrototype` crash). I also switch to a sentiment-extraction checkpoint (Tweet sentiment extraction fine-tuned RoBERTa) while keeping the exact same QA span-extraction pipeline, which should increase the Jaccard score substantially toward your target. Finally, I keep the submission-writing logic intact and ensure it always produces a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
df_test.head(2)



## === cell 2
sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
sub_df.head(2)



## === cell 3
sub_df.shape



## === cell 4
import tensorflow as tf
from transformers import AutoTokenizer

TOKENIZER_NAME = "deepset/roberta-base-squad2"
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
from transformers import TFAutoModelForQuestionAnswering

MODEL_NAME = "deepset/roberta-base-squad2"
try:
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        MODEL_NAME, local_files_only=True
    )
except Exception:
    model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)



## === cell 6
from datasets import Dataset

df_test.reset_index(drop=True, inplace=True)

test_data = Dataset.from_pandas(df_test)
test_data



## === cell 7
MAX_LENGTH = 73


def post_porocess_data(examples):
    questions = examples["sentiment"]
    context = examples["text"]
    inputs = tokenizer(
        questions,
        context,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )

    for i in range(len(inputs["input_ids"])):
        offset = inputs["offset_mapping"][i]
        sequence_ids = inputs.sequence_ids(i)
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]
    return inputs




## === cell 8
processed_test_data = test_data.map(post_porocess_data, batched=True)
processed_test_data



## === cell 9
tf_test_dataset = processed_test_data.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    shuffle=False,
    batch_size=16,
)



## === cell 10
outputs = model.predict(tf_test_dataset, verbose=0)

if isinstance(outputs, dict):
    start_logits = outputs["start_logits"]
    end_logits = outputs["end_logits"]
else:
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits



## === cell 11
n_best = 20


def predict_answers(inputs):
    predicted_answer = []
    for i in range(len(inputs["offset_mapping"])):
        start_logit = inputs["start_logits"][i]
        end_logit = inputs["end_logits"][i]
        context = inputs["text"][i]
        offset = inputs["offset_mapping"][i]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()

        found = False
        best_answer = None

        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(offset) or end_index >= len(offset):
                    continue
                if offset[start_index] is None or offset[end_index] is None:
                    continue
                if end_index < start_index:
                    continue
                found = True
                best_answer = context[offset[start_index][0] : offset[end_index][1]]
                break
            if found:
                break

        if not found or best_answer is None or len(best_answer.strip()) == 0:
            best_answer = context

        predicted_answer.append(best_answer)

    return {"predicted_answer": predicted_answer}




## === cell 12
processed_test_data.set_format("pandas")



## === cell 13
processed_test_df = processed_test_data[:]
processed_test_df.head(2)



## === cell 14
processed_test_df["start_logits"] = start_logits.tolist()
processed_test_df["end_logits"] = end_logits.tolist()
processed_test_df.shape



## === cell 15
processed_test_df["text"] = df_test.text
processed_test_df.head(2)



## === cell 16
final_test_data = Dataset.from_pandas(processed_test_df)
final_test_data



## === cell 17
final_test_data = final_test_data.map(predict_answers, batched=True)
final_test_data



## === cell 18
sub_df["selected_text"] = final_test_data["predicted_answer"]
sub_df.to_csv("submission.csv", index=False)

print("File submitted successfully. Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
