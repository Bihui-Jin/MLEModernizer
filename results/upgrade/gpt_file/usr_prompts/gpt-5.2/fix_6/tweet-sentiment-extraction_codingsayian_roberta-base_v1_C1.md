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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

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
import google.protobuf  # noqa: F401

import tensorflow as tf

try:
    import tf_keras  # noqa: F401
except Exception:
    pass

from transformers import AutoTokenizer

LOCAL_MODEL_DIR = "/kaggle/input/roberta-base-finetuned-tweet-sentiment-extraction"
HF_MODEL_NAME = "mrm8488/roberta-base-finetuned-tweet-sentiment-extraction"
FALLBACK_MODEL_NAME = "roberta-base"

tokenizer = None
load_errors = []

if os.path.isdir(LOCAL_MODEL_DIR):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_DIR, local_files_only=True
        )
    except Exception as e:
        load_errors.append(("LOCAL_MODEL_DIR", repr(e)))

if tokenizer is None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, local_files_only=True)
    except Exception as e:
        load_errors.append(("HF_MODEL_NAME local_files_only", repr(e)))

if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained(
        FALLBACK_MODEL_NAME, local_files_only=True
    )

print("Loaded tokenizer:", getattr(tokenizer, "name_or_path", "unknown"))
if load_errors:
    print("Tokenizer load fallbacks tried (first 3):", load_errors[:3])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
from transformers import TFAutoModelForQuestionAnswering

model = None
load_errors = []

if os.path.isdir(LOCAL_MODEL_DIR):
    try:
        model = TFAutoModelForQuestionAnswering.from_pretrained(
            LOCAL_MODEL_DIR, local_files_only=True
        )
    except Exception as e:
        load_errors.append(("LOCAL_MODEL_DIR", repr(e)))

if model is None:
    try:
        model = TFAutoModelForQuestionAnswering.from_pretrained(
            HF_MODEL_NAME, local_files_only=True
        )
    except Exception as e:
        load_errors.append(("HF_MODEL_NAME local_files_only", repr(e)))

if model is None:
    model = TFAutoModelForQuestionAnswering.from_pretrained(
        FALLBACK_MODEL_NAME, local_files_only=True
    )

print("Loaded model:", getattr(model, "name_or_path", "unknown"))
if load_errors:
    print("Model load fallbacks tried (first 3):", load_errors[:3])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/467242796.py in <cell line: 0>()
     22 if model is None:
     23     # Offline-safe fallback: base RoBERTa QA head (not fine-tuned, but ensures a valid submission).
---> 24     model = TFAutoModelForQuestionAnswering.from_pretrained(
     25         FALLBACK_MODEL_NAME, local_files_only=True
     26     )

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py in from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    545                 _ = kwargs.pop("quantization_config")
    546 
--> 547             config, kwargs = AutoConfig.from_pretrained(
    548                 pretrained_model_name_or_path,
    549                 return_unused_kwargs=True,

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1195         code_revision = kwargs.pop("code_revision", None)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
   1199         has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

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



## --- ERROR in cell 8, traceback:
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

/tmp/ipykernel_11/4241274052.py in post_porocess_data(examples)
      5     questions = examples["sentiment"]
      6     context = examples["text"]
----> 7     inputs = tokenizer(
      8         questions,
      9         context,

TypeError: 'NoneType' object is not callable

## === cell 9
tf_test_dataset = processed_test_data.to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    shuffle=False,
    batch_size=16,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2869352155.py in <cell line: 0>()
----> 1 tf_test_dataset = processed_test_data.to_tf_dataset(
      2     columns=["input_ids", "attention_mask"],
      3     shuffle=False,
      4     batch_size=16,
      5 )

NameError: name 'processed_test_data' is not defined

## === cell 10
outputs = model.predict(tf_test_dataset, verbose=0)

start_logits = None
end_logits = None

if isinstance(outputs, dict):
    start_logits = outputs.get("start_logits", None)
    end_logits = outputs.get("end_logits", None)
elif hasattr(outputs, "start_logits") and hasattr(outputs, "end_logits"):
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
    start_logits, end_logits = outputs[0], outputs[1]
else:
    raise TypeError(f"Unexpected model.predict output type: {type(outputs)}")

if start_logits is None or end_logits is None:
    raise ValueError("Could not extract start_logits/end_logits from model outputs.")

start_logits = np.asarray(start_logits)
end_logits = np.asarray(end_logits)

start_logits.shape, end_logits.shape



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2706648717.py in <cell line: 0>()
      1 # Fix: robustly extract logits from TF/Keras predict output across transformers versions.
----> 2 outputs = model.predict(tf_test_dataset, verbose=0)
      3 
      4 start_logits = None
      5 end_logits = None

AttributeError: 'NoneType' object has no attribute 'predict'

## === cell 11
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




## === cell 12
processed_test_data.set_format("pandas")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/375194404.py in <cell line: 0>()
----> 1 processed_test_data.set_format("pandas")
      2 

NameError: name 'processed_test_data' is not defined

## === cell 13
processed_test_df = processed_test_data[:]
processed_test_df.head(2)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2124900806.py in <cell line: 0>()
----> 1 processed_test_df = processed_test_data[:]
      2 processed_test_df.head(2)
      3 

NameError: name 'processed_test_data' is not defined

## === cell 14
processed_test_df["start_logits"] = start_logits.tolist()
processed_test_df["end_logits"] = end_logits.tolist()
processed_test_df.shape



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2427158753.py in <cell line: 0>()
----> 1 processed_test_df["start_logits"] = start_logits.tolist()
      2 processed_test_df["end_logits"] = end_logits.tolist()
      3 processed_test_df.shape
      4 

NameError: name 'start_logits' is not defined

## === cell 15
processed_test_df["text"] = df_test.text
processed_test_df["sentiment"] = df_test.sentiment
processed_test_df.head(2)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2650867125.py in <cell line: 0>()
----> 1 processed_test_df["text"] = df_test.text
      2 processed_test_df["sentiment"] = df_test.sentiment
      3 processed_test_df.head(2)
      4 

NameError: name 'processed_test_df' is not defined

## === cell 16
final_test_data = Dataset.from_pandas(processed_test_df)
final_test_data



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/321394903.py in <cell line: 0>()
----> 1 final_test_data = Dataset.from_pandas(processed_test_df)
      2 final_test_data
      3 

NameError: name 'processed_test_df' is not defined

## === cell 17
final_test_data = final_test_data.map(predict_answers, batched=True)
final_test_data



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2729544137.py in <cell line: 0>()
----> 1 final_test_data = final_test_data.map(predict_answers, batched=True)
      2 final_test_data
      3 

NameError: name 'final_test_data' is not defined

## === cell 18
sub_df["selected_text"] = final_test_data["predicted_answer"]

sub_df = sub_df[["textID", "selected_text"]]
assert len(sub_df) == len(df_test), "Submission/test row count mismatch"

sub_df.to_csv("submission.csv", index=False)

print("File submitted successfully. Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2934761955.py in <cell line: 0>()
----> 1 sub_df["selected_text"] = final_test_data["predicted_answer"]
      2 
      3 sub_df = sub_df[["textID", "selected_text"]]
      4 assert len(sub_df) == len(df_test), "Submission/test row count mismatch"
      5 

NameError: name 'final_test_data' is not defined
