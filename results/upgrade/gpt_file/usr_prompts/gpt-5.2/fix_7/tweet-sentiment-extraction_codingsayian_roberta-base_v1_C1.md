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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.29108) has done: 'The crashes come from trying to load a tokenizer/model from non-existent local Kaggle input folders (`../input/robertatokenizer/` and `../input/robertamodel`), which leaves `tokenizer`/`model` undefined and breaks all downstream cells. I minimally fix this by (1) using the standard `roberta-base` tokenizer/model from Hugging Face cache (offline-safe on Kaggle) with a local-only fallback, and (2) ensuring TensorFlow is imported (needed for `to_tf_dataset`/`model.predict`). I also fix a small logic bug in `predict_answers` where `answer` could be referenced before assignment if no valid span is found, by falling back to the full tweet text. These changes preserve the original QA-span extraction approach and produce a valid `submission.csv`.'
- What this solution (achieved 0.30638) has done: 'I fix the runtime error caused by an incompatible `protobuf`/`transformers` import path in this Kaggle environment by pinning protobuf to the pure-Python implementation via an environment variable before importing `transformers` (this resolves the `MessageFactory.GetPrototype` crash). I also switch to a sentiment-extraction checkpoint (Tweet sentiment extraction fine-tuned RoBERTa) while keeping the exact same QA span-extraction pipeline, which should increase the Jaccard score substantially toward your target. Finally, I keep the submission-writing logic intact and ensure it always produces a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.52465) has done: 'I fix the immediate crash in `transformers` caused by the protobuf C++ implementation by forcing the pure-Python protobuf backend *and* ensuring this environment variable is set before Python imports protobuf/transformers (and restarting isn’t needed in Kaggle scripts). I also make the input CSV paths consistent with the provided filesystem (`/kaggle/input/...`) so the notebook works in Kaggle reliably. Finally, I keep your QA-span extraction logic intact but add a minimal, score-improving fallback for neutral sentiment (return full text) which is standard for this competition and doesn’t change the core approach.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
df_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
df_test.head(2)



## === cell 2
sub_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
sub_df.head(2)



## === cell 3
sub_df.shape



## === cell 4
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def find_local_model_dir(search_root="/kaggle/input"):
    """
    Try to find a directory containing a Hugging Face model (config.json + weights).
    Prioritize directories that look like tweet sentiment extraction fine-tunes.
    """
    root = Path(search_root)
    candidates = []
    for p in root.rglob("config.json"):
        d = p.parent
        has_weights = any(
            (d / w).exists()
            for w in ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]
        )
        if not has_weights:
            continue
        score = 0
        name = str(d).lower()
        if "tweet" in name:
            score += 3
        if "sentiment" in name:
            score += 3
        if "extraction" in name:
            score += 2
        if "roberta" in name:
            score += 1
        candidates.append((score, str(d)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


LOCAL_MODEL_DIR = find_local_model_dir("/kaggle/input")
FALLBACK_MODEL_DIRS = [
    "/kaggle/input/roberta-base-finetuned-tweet-sentiment-extraction",
    "/kaggle/input/tweet-sentiment-extraction",  # unlikely to contain model, but harmless
]
FALLBACK_MODEL_NAME = "roberta-base"

tokenizer = None
model = None
load_notes = []


def try_load_from_dir(d):
    global tokenizer, model
    if d is None or (not os.path.isdir(d)):
        return False
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            d, local_files_only=True, use_fast=True
        )
        model = AutoModelForQuestionAnswering.from_pretrained(d, local_files_only=True)
        return True
    except Exception as e:
        load_notes.append((d, repr(e)))
        tokenizer = None
        model = None
        return False


loaded = False
loaded = try_load_from_dir(LOCAL_MODEL_DIR)
if not loaded:
    for d in FALLBACK_MODEL_DIRS:
        if try_load_from_dir(d):
            loaded = True
            break

if not loaded:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            FALLBACK_MODEL_NAME, local_files_only=True, use_fast=True
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            FALLBACK_MODEL_NAME, local_files_only=True
        )
        loaded = True
        load_notes.append(("FALLBACK_MODEL_NAME(cache)", "loaded"))
    except Exception as e:
        load_notes.append(("FALLBACK_MODEL_NAME(cache)", repr(e)))
        tokenizer = None
        model = None

print("Local model dir picked:", LOCAL_MODEL_DIR)
print("Loaded tokenizer:", getattr(tokenizer, "name_or_path", None))
print("Loaded model:", getattr(model, "name_or_path", None))
if load_notes:
    print("Load notes (first 5):", load_notes[:5])

if tokenizer is None or model is None:
    raise RuntimeError(
        "Could not load any tokenizer/model offline. "
        "Please add a Hugging Face QA checkpoint as a Kaggle dataset/input containing config.json + weights."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/830311396.py in <cell line: 0>()
    104 # Hard fail early if nothing could be loaded; otherwise downstream would crash.
    105 if tokenizer is None or model is None:
--> 106     raise RuntimeError(
    107         "Could not load any tokenizer/model offline. "
    108         "Please add a Hugging Face QA checkpoint as a Kaggle dataset/input containing config.json + weights."

RuntimeError: Could not load any tokenizer/model offline. Please add a Hugging Face QA checkpoint as a Kaggle dataset/input containing config.json + weights.

## === cell 5
from datasets import Dataset

df_test = df_test.reset_index(drop=True)
test_data = Dataset.from_pandas(df_test)
test_data



## === cell 6
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




## === cell 7
processed_test_data = test_data.map(post_porocess_data, batched=True)
processed_test_data



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/731138799.py in <cell line: 0>()
----> 1 processed_test_data = test_data.map(post_porocess_data, batched=True)
      2 processed_test_data
      3 

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3339                 else:
   3340                     for unprocessed_kwargs in unprocessed_kwargs_per_job:
-> 3341                         for rank, done, content in Dataset._map_single(**unprocessed_kwargs):
   3342                             check_if_shard_done(rank, done, content)
   3343 

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in _map_single(shard, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, new_fingerprint, rank, offset, try_original_type)
   3695                 else:
   3696                     _time = time.time()
-> 3697                     for i, batch in iter_outputs(shard_iterable):
   3698                         num_examples_in_batch = len(i)
   3699                         if update_data:

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in iter_outputs(shard_iterable)
   3645             else:
   3646                 for i, example in shard_iterable:
-> 3647                     yield i, apply_function(example, i, offset=offset)
   3648 
   3649         num_examples_progress_update = 0

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in apply_function(pa_inputs, indices, offset)
   3568             """Utility to apply the function on a selection of columns."""
   3569             inputs, fn_args, additional_args, fn_kwargs = prepare_inputs(pa_inputs, indices, offset=offset)
-> 3570             processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
   3571             return prepare_outputs(pa_inputs, inputs, processed_inputs)
   3572 

/tmp/ipykernel_11/3553321059.py in post_porocess_data(examples)
      5     questions = examples["sentiment"]
      6     context = examples["text"]
----> 7     inputs = tokenizer(
      8         questions,
      9         context,

TypeError: 'NoneType' object is not callable

## === cell 8
from torch.utils.data import DataLoader

cols_to_keep = ["input_ids", "attention_mask", "offset_mapping"]
processed_test_data = processed_test_data.remove_columns(
    [c for c in processed_test_data.column_names if c not in set(cols_to_keep)]
)

processed_test_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

loader = DataLoader(processed_test_data, batch_size=32, shuffle=False)

all_start = []
all_end = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        all_start.append(out.start_logits.detach().cpu().numpy())
        all_end.append(out.end_logits.detach().cpu().numpy())

start_logits = np.concatenate(all_start, axis=0)
end_logits = np.concatenate(all_end, axis=0)

print("Logits shapes:", start_logits.shape, end_logits.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1742154167.py in <cell line: 0>()
      4 # Keep only what we need for inference; keep offset mapping for later span extraction.
      5 cols_to_keep = ["input_ids", "attention_mask", "offset_mapping"]
----> 6 processed_test_data = processed_test_data.remove_columns(
      7     [c for c in processed_test_data.column_names if c not in set(cols_to_keep)]
      8 )

NameError: name 'processed_test_data' is not defined

## === cell 9
n_best = 20


def predict_answers(inputs):
    predicted_answer = []
    for i in range(len(inputs["offset_mapping"])):
        start_logit = inputs["start_logits"][i]
        end_logit = inputs["end_logits"][i]
        context = inputs["text"][i]
        offset = inputs["offset_mapping"][i]

        sent = inputs["sentiment"][i] if "sentiment" in inputs else None
        if sent == "neutral":
            predicted_answer.append(context)
            continue

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




## === cell 10
processed_test_data.set_format(None)  # back to python lists for offsets

processed_test_df = processed_test_data.to_pandas()
processed_test_df["start_logits"] = start_logits.tolist()
processed_test_df["end_logits"] = end_logits.tolist()
processed_test_df["text"] = df_test["text"].values
processed_test_df["sentiment"] = df_test["sentiment"].values

from datasets import Dataset

final_test_data = Dataset.from_pandas(processed_test_df)
final_test_data



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3856286631.py in <cell line: 0>()
      1 # Fix: rebuild a dataframe that contains offsets + logits + original text/sentiment for prediction.
----> 2 processed_test_data.set_format(None)  # back to python lists for offsets
      3 
      4 processed_test_df = processed_test_data.to_pandas()
      5 processed_test_df["start_logits"] = start_logits.tolist()

NameError: name 'processed_test_data' is not defined

## === cell 11
final_test_data = final_test_data.map(predict_answers, batched=True)
final_test_data



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2729544137.py in <cell line: 0>()
----> 1 final_test_data = final_test_data.map(predict_answers, batched=True)
      2 final_test_data
      3 

NameError: name 'final_test_data' is not defined

## === cell 12
sub_df["selected_text"] = final_test_data["predicted_answer"]
sub_df = sub_df[["textID", "selected_text"]]
assert len(sub_df) == len(df_test), "Submission/test row count mismatch"

sub_df.to_csv("submission.csv", index=False)
print("File submitted successfully. Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/715276201.py in <cell line: 0>()
      1 # Write submission with required columns; pandas will quote as needed in CSV.
----> 2 sub_df["selected_text"] = final_test_data["predicted_answer"]
      3 sub_df = sub_df[["textID", "selected_text"]]
      4 assert len(sub_df) == len(df_test), "Submission/test row count mismatch"
      5 

NameError: name 'final_test_data' is not defined
