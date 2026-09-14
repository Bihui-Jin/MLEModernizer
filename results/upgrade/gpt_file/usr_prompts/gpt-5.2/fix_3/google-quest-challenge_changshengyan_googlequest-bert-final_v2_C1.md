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

0.3475238241448931

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import warnings

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.stats import spearmanr

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

import tensorflow.keras.backend as K

from transformers import BertTokenizer, TFBertModel, BertConfig

warnings.filterwarnings("ignore")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
max_sequence_length = 380  # Higher will cause memory issues
n_epoch = 3  # kept for compatibility; inference-only script
learning_rate = 2e-5
n_fold = 5
batch_size = 8
dropout_rate = 0.1




## === cell 2
model_version = "bertv11-3e/best_model_2.h5"

DATA_PATH_CANDIDATES = [
    "/kaggle/input/google-quest-challenge",
    "/kaggle/input/google-quest-challenge/google-quest-challenge",
    "/kaggle/data/google-quest-challenge",
    "/kaggle/data/google-quest-challenge/google-quest-challenge",
    "../input/google-quest-challenge",
]
DATA_PATH = next((p for p in DATA_PATH_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError(
        f"Could not find competition data folder in: {DATA_PATH_CANDIDATES}"
    )

print("Using DATA_PATH:", DATA_PATH)

BERT_ID = "bert-base-uncased"
print("Using BERT_ID:", BERT_ID)




## === cell 3
try:
    tokenizer = BertTokenizer.from_pretrained(BERT_ID, local_files_only=True)
except Exception as e:
    raise RuntimeError(
        "Failed to load 'bert-base-uncased' tokenizer from local cache. "
        "This notebook requires the model to be available in the Kaggle image cache. "
        "If it's missing, add a Kaggle Dataset containing the BERT files or switch to an available local model."
    ) from e

df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

input_columns = ["question_title", "question_body", "answer"]

df_sample = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
output_labels = list(df_sample.columns[1:])

print("n_train:", len(df_train), "n_test:", len(df_test))
print("n_targets:", len(output_labels))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1387583657.py in <cell line: 0>()
      3 try:
----> 4     tokenizer = BertTokenizer.from_pretrained(BERT_ID, local_files_only=True)
      5 except Exception as e:

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   2013 
-> 2014         return cls._from_pretrained(
   2015             resolved_vocab_files,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2259         try:
-> 2260             tokenizer = cls(*init_inputs, **init_kwargs)
   2261         except import_protobuf_decode_error():

/usr/local/lib/python3.11/dist-packages/transformers/models/bert/tokenization_bert.py in __init__(self, vocab_file, do_lower_case, do_basic_tokenize, never_split, unk_token, sep_token, pad_token, cls_token, mask_token, tokenize_chinese_chars, strip_accents, clean_up_tokenization_spaces, **kwargs)
    113     ):
--> 114         if not os.path.isfile(vocab_file):
    115             raise ValueError(

/usr/lib/python3.11/genericpath.py in isfile(path)

TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1387583657.py in <cell line: 0>()
      4     tokenizer = BertTokenizer.from_pretrained(BERT_ID, local_files_only=True)
      5 except Exception as e:
----> 6     raise RuntimeError(
      7         "Failed to load 'bert-base-uncased' tokenizer from local cache. "
      8         "This notebook requires the model to be available in the Kaggle image cache. "

RuntimeError: Failed to load 'bert-base-uncased' tokenizer from local cache. This notebook requires the model to be available in the Kaggle image cache. If it's missing, add a Kaggle Dataset containing the BERT files or switch to an available local model.

## === cell 4
def process_sequence(str1, str2, length):
    """
    Process sequence or sequence pair into ids, masks and segments.
    """
    inputs = tokenizer.encode_plus(
        str1 if str1 is not None else "",
        str2,
        add_special_tokens=True,
        max_length=length,
        truncation=True,
        padding=False,
        return_token_type_ids=True,
        return_attention_mask=False,
    )

    ids = inputs["input_ids"]
    seg = inputs.get("token_type_ids", [0] * len(ids))
    mask = [1] * len(ids)

    padding_length = length - len(ids)
    if padding_length > 0:
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        seg = seg + ([0] * padding_length)
    else:
        ids = ids[:length]
        mask = mask[:length]
        seg = seg[:length]

    return ids, mask, seg


def convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """
    Preprocess text input into tokens, then encode them into ids, masks and segments as input for BERT.
    """
    title = "" if pd.isna(title) else str(title)
    question = "" if pd.isna(question) else str(question)
    answer = "" if pd.isna(answer) else str(answer)

    id_q, mask_q, segment_q = process_sequence(
        title + " " + question, None, max_sequence_length
    )
    id_a, mask_a, segment_a = process_sequence(answer, None, max_sequence_length)

    return id_q, mask_q, segment_q, id_a, mask_a, segment_a




## === cell 5
def compute_input(df, columns, tokenizer, max_sequence_length):
    ids_q, masks_q, segments_q = [], [], []
    ids_a, masks_a, segments_a = [], [], []

    for _, instance in tqdm(df[columns].iterrows(), total=len(df)):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        id_q, mask_q, segment_q, id_a, mask_a, segment_a = convert_to_bert_inputs(
            t, q, a, tokenizer, max_sequence_length
        )

        ids_q.append(id_q)
        masks_q.append(mask_q)
        segments_q.append(segment_q)

        ids_a.append(id_a)
        masks_a.append(mask_a)
        segments_a.append(segment_a)

    return [
        np.asarray(ids_q, dtype=np.int32),
        np.asarray(masks_q, dtype=np.int32),
        np.asarray(segments_q, dtype=np.int32),
        np.asarray(ids_a, dtype=np.int32),
        np.asarray(masks_a, dtype=np.int32),
        np.asarray(segments_a, dtype=np.int32),
    ]


def compute_output(df, columns):
    return np.asarray(df[columns], dtype=np.float32)




## === cell 6
def create_model():
    q_id = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)

    q_mask = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)

    q_seg = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)
    a_seg = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)

    config = BertConfig.from_pretrained(BERT_ID, local_files_only=True)
    config.output_hidden_states = False

    bert_model = TFBertModel.from_pretrained(
        BERT_ID, config=config, local_files_only=True
    )

    q_emb = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_seg)[0]
    a_emb = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_seg)[0]

    q = tf.keras.layers.GlobalAveragePooling1D()(q_emb)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_emb)

    x = tf.keras.layers.Concatenate()([q, a])

    x = tf.keras.layers.Dense(1500)(x)
    x = tf.keras.layers.Dense(1500)(x)

    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(30, activation="sigmoid")(x)

    model = tf.keras.models.Model(
        inputs=[q_id, q_mask, q_seg, a_id, a_mask, a_seg], outputs=x
    )
    return model


def compute_rho(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)




## === cell 7
test_preds = []
K.clear_session()

model = create_model()

weights_path_candidates = [
    os.path.join("/kaggle/input", model_version),
    os.path.join("../input", model_version),
    os.path.join("/kaggle/data", model_version),
]
weights_path = next((p for p in weights_path_candidates if os.path.exists(p)), None)
if weights_path is None:
    raise FileNotFoundError(
        f"Could not find model weights for model_version={model_version}. "
        f"Tried: {weights_path_candidates}"
    )

model.load_weights(weights_path)

test_inputs = compute_input(df_test, input_columns, tokenizer, max_sequence_length)
pred = model.predict(test_inputs, batch_size=batch_size, verbose=1)
test_preds.append(pred)

df_submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

avg_pred = np.average(test_preds, axis=0)
avg_pred = np.clip(avg_pred, 0.0, 1.0)

if avg_pred.shape != (len(df_submission), len(output_labels)):
    raise ValueError(
        f"Prediction shape {avg_pred.shape} does not match submission shape "
        f"({len(df_submission)}, {len(output_labels)})."
    )

df_submission.iloc[:, 1:] = avg_pred
df_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", df_submission.shape)
print(df_submission.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))

## --- ERROR in cell 7, traceback:
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
/tmp/ipykernel_11/1655389233.py in <cell line: 0>()
      3 K.clear_session()
      4 
----> 5 model = create_model()
      6 
      7 # Fix: robustly find weights path in Kaggle inputs

/tmp/ipykernel_11/3343985117.py in create_model()
     10 
     11     # Fix: load config/model from HF cache by model id (offline)
---> 12     config = BertConfig.from_pretrained(BERT_ID, local_files_only=True)
     13     config.output_hidden_states = False
     14 

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, **kwargs)
    566         cls._set_token_in_kwargs(kwargs, token)
    567 
--> 568         config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    569         if cls.base_config_key and cls.base_config_key in config_dict:
    570             config_dict = config_dict[cls.base_config_key]

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
