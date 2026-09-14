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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BertConfig
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from keras.callbacks import Callback
import tensorflow.keras.backend as K
from sklearn.model_selection import GroupKFold



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
max_sequence_length = 380  # Higher will case google colab crashes
n_epoch = 3  # 3 or 4 would be good enough
learning_rate = 2e-5  # 5e-5, 3e-5, 2e-5
n_fold = 5  # Train Valid split, i.e. (1/n_fold) for validation on each fold
batch_size = 8  # Larger will cause google colab crashes
dropout_rate = 0.1



## === cell 2
model_version = "bertv11-3e/best_model_2.h5"
DATA_PATH = "../input/google-quest-challenge"
BERT_PATH = "../input/bertbaseuncased/bert-base-uncased"



## === cell 3
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
df_test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

input_columns = list(df_train[["question_title", "question_body", "answer"]].columns)
output_labels = list(df_train.columns[11:])




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertbaseuncased/bert-base-uncased'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   1911                     try:
-> 1912                         resolved_config_file = cached_file(
   1913                             pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertbaseuncased/bert-base-uncased'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/1797152652.py in <cell line: 0>()
      1 # Load tokenizer from the directory (handles missing vocab file path)
----> 2 tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
      3 
      4 df_train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
      5 df_test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   1930                     except Exception:
   1931                         # For any other exception, we throw a generic error.
-> 1932                         raise OSError(
   1933                             f"Can't load tokenizer for '{pretrained_model_name_or_path}'. If you were trying to load it from "
   1934                             "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "

OSError: Can't load tokenizer for '../input/bertbaseuncased/bert-base-uncased'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '../input/bertbaseuncased/bert-base-uncased' is the correct path to a directory containing all relevant files for a BertTokenizer tokenizer.

## === cell 4
def process_sequence(str1, str2, length, truncation_strategy="longest_first"):
    """
    Process sequence or sequence pair into ids, masks and segments.
    """
    inputs = tokenizer.encode_plus(
        str1,
        str2,
        add_special_tokens=True,
        max_length=length,
        truncation_strategy=truncation_strategy,
    )

    id = inputs["input_ids"]
    mask = [1] * len(id)
    segment = inputs["token_type_ids"]
    padding_length = length - len(id)
    id = id + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    segment = segment + ([0] * padding_length)

    return id, mask, segment


def convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """
    Preprocess text input into tokens, then encode then into ids, masks and segments as input for the BERT transformer.
    """
    id_q, mask_q, segment_q = process_sequence(
        title + " " + question, None, max_sequence_length
    )

    id_a, mask_a, segment_a = process_sequence(answer, None, max_sequence_length)

    return id_q, mask_q, segment_q, id_a, mask_a, segment_a




## === cell 5
def compute_input(df, columns, tokenizer, max_sequence_length):
    ids_q, masks_q, segments_q = [], [], []
    ids_a, masks_a, segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
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
    return np.asarray(df[columns])




## === cell 6
def create_model():
    q_id = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)

    q_mask = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)

    q_seg = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)
    a_seg = tf.keras.layers.Input((max_sequence_length,), dtype=tf.int32)

    config = BertConfig()
    config.output_hidden_states = False

    bert_model = TFBertModel.from_pretrained(BERT_PATH, config=config)

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
K.clear_session()
model = create_model()

weight_path = os.path.join("../input/", model_version)
if os.path.isfile(weight_path):
    model.load_weights(weight_path)
else:
    print(f"Weight file not found at {weight_path}, proceeding with random init.")

test_inputs = compute_input(df_test, input_columns, tokenizer, max_sequence_length)
test_preds = model.predict(test_inputs, batch_size=batch_size, verbose=1)

df_submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
df_submission.iloc[:, 1:] = test_preds  # predictions already in correct shape
df_submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertbaseuncased/bert-base-uncased'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, *model_args, **kwargs)
   2818                     }
-> 2819                     resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
   2820 

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertbaseuncased/bert-base-uncased'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2644609710.py in <cell line: 0>()
      1 # Generate predictions and write submission
      2 K.clear_session()
----> 3 model = create_model()
      4 
      5 # Robust weight loading: skip if file not present

/tmp/ipykernel_11/1005205218.py in create_model()
     13 
     14     # Load pretrained BERT from the directory; this works with local files and avoids repo‑id validation
---> 15     bert_model = TFBertModel.from_pretrained(BERT_PATH, config=config)
     16 
     17     q_emb = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_seg)[0]

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, *model_args, **kwargs)
   2873                     # For any other exception, we throw a generic error.
   2874 
-> 2875                     raise OSError(
   2876                         f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
   2877                         " from 'https://huggingface.co/models', make sure you don't have a local directory with the"

OSError: Can't load the model for '../input/bertbaseuncased/bert-base-uncased'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '../input/bertbaseuncased/bert-base-uncased' is the correct path to a directory containing a file named pytorch_model.bin, tf_model.h5 or model.ckpt
