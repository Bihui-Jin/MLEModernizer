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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tokenizers==0.21.2
transformers==4.53.3

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

0.2441508446366312

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
DIR = '/kaggle/input/google-quest-challenge/'
BATCH_SIZE = 64

## === cell 2
import transformers
import tokenizers
import tensorflow as tf 
import pandas as pd
import functools
import numpy as np
import os
from scipy.stats import rankdata, spearmanr

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3

encoder = transformers.TFBertModel.from_pretrained('/kaggle/input/model-1-quest/kaggle/working/models')
tokenizer = transformers.BertTokenizerFast.from_pretrained('/kaggle/input/model-1-quest/kaggle/working/models')

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/model-1-quest/kaggle/working/models'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/model-1-quest/kaggle/working/models'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_10/3218166563.py in <cell line: 0>()
      6 # encoder.save_pretrained('/kaggle/working/models')
      7 # tokenizer.save_pretrained('/kaggle/working/models')
----> 8 encoder = transformers.TFBertModel.from_pretrained('/kaggle/input/model-1-quest/kaggle/working/models')
      9 tokenizer = transformers.BertTokenizerFast.from_pretrained('/kaggle/input/model-1-quest/kaggle/working/models')

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, *model_args, **kwargs)
   2708         if not isinstance(config, PretrainedConfig):
   2709             config_path = config if config is not None else pretrained_model_name_or_path
-> 2710             config, model_kwargs = cls.config_class.from_pretrained(
   2711                 config_path,
   2712                 cache_dir=cache_dir,

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
    688             except Exception:
    689                 # For any other exception, we throw a generic error.
--> 690                 raise OSError(
    691                     f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
    692                     " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"

OSError: Can't load the configuration of '/kaggle/input/model-1-quest/kaggle/working/models'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/kaggle/input/model-1-quest/kaggle/working/models' is the correct path to a directory containing a config.json file

## === cell 5
def preprocess_data(df, offset = 0):
    title = df['question_title'].tolist()
    body = df['question_body'].tolist()
    answer = df['answer'].tolist()
    
    title_tokens = tokenizer.batch_encode_plus(title, return_tensors = 'tf', pad_to_max_length = True, add_special_tokens = True, 
                                return_attention_masks = True, return_token_type_ids = False)
    body_tokens = tokenizer.batch_encode_plus(body, return_tensors = 'tf', pad_to_max_length = True, add_special_tokens = True, 
                                    return_attention_masks = True, return_token_type_ids = False, max_length = 171, )
    answer_tokens = tokenizer.batch_encode_plus(answer, return_tensors = 'tf', pad_to_max_length = True, add_special_tokens = True, 
                                    return_attention_masks = True, return_token_type_ids = False, max_length = 233,)
    
    max_title_tokens = title_tokens['input_ids'].shape[1]
    max_body_tokens = body_tokens['input_ids'].shape[1]-offset
    max_answer_tokens = answer_tokens['input_ids'].shape[1]-offset
    
    
    t = {}
    t['input_ids'] = \
    tf.concat([title_tokens['input_ids'][:, :max_title_tokens], body_tokens['input_ids'][:, :max_body_tokens], 
               answer_tokens['input_ids'][:, :max_answer_tokens]], axis = -1)
    t['attention_mask'] = \
    tf.concat([title_tokens['attention_mask'][:, :max_title_tokens], body_tokens['attention_mask'][:, :max_body_tokens], 
               answer_tokens['attention_mask'][:, :max_answer_tokens]], axis = -1)
    
    return t

## === cell 6
train_df = pd.read_csv(DIR+'/train.csv')
test_df = pd.read_csv(DIR+'/test.csv')

train_t = preprocess_data(train_df, 4)
test_t = preprocess_data(test_df)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1249090444.py in <cell line: 0>()
      2 test_df = pd.read_csv(DIR+'/test.csv')
      3 
----> 4 train_t = preprocess_data(train_df, 4)
      5 test_t = preprocess_data(test_df)

/tmp/ipykernel_10/1835918886.py in preprocess_data(df, offset)
      4     answer = df['answer'].tolist()
      5 
----> 6     title_tokens = tokenizer.batch_encode_plus(title, return_tensors = 'tf', pad_to_max_length = True, add_special_tokens = True, 
      7                                 return_attention_masks = True, return_token_type_ids = False)
      8     body_tokens = tokenizer.batch_encode_plus(body, return_tensors = 'tf', pad_to_max_length = True, add_special_tokens = True, 

NameError: name 'tokenizer' is not defined

## === cell 7
labels = train_df.iloc[:, -30:]


## === cell 8
input_ids = tf.keras.layers.Input(shape = (461,), dtype = tf.int32)
input_masks = tf.keras.layers.Input(shape = (461,), dtype = tf.float32)

bert_output = encoder({'input_ids':input_ids, 'attention_masks':input_masks}, training = False)[0][:, 0, :]
model_1 = tf.keras.Model(inputs = [input_ids, input_masks], outputs = [bert_output])

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3692231879.py in <cell line: 0>()
      2 input_masks = tf.keras.layers.Input(shape = (461,), dtype = tf.float32)
      3 
----> 4 bert_output = encoder({'input_ids':input_ids, 'attention_masks':input_masks}, training = False)[0][:, 0, :]
      5 model_1 = tf.keras.Model(inputs = [input_ids, input_masks], outputs = [bert_output])

NameError: name 'encoder' is not defined

## === cell 9
train_encoded_inputs_tensor = model_1.predict([train_t['input_ids'], train_t['attention_mask']], batch_size = 64)
test_encoded_inputs_tensor = model_1.predict([test_t['input_ids'], test_t['attention_mask']], batch_size = 64)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2373665841.py in <cell line: 0>()
----> 1 train_encoded_inputs_tensor = model_1.predict([train_t['input_ids'], train_t['attention_mask']], batch_size = 64)
      2 test_encoded_inputs_tensor = model_1.predict([test_t['input_ids'], test_t['attention_mask']], batch_size = 64)

NameError: name 'model_1' is not defined

## === cell 10
columns = list(train_df.columns[-30:])

## === cell 11
encoded_inputs = tf.keras.layers.Input(shape = (768))
dropout = tf.keras.layers.Dropout(rate = 0.2)
drop_encoded_inputs = dropout(encoded_inputs)
dense_1 = tf.keras.layers.Dense(512, activation = 'elu')(drop_encoded_inputs)
dense_2 = tf.keras.layers.Dense(30, activation = 'sigmoid')(dense_1)
model_2 = tf.keras.Model(inputs = [encoded_inputs], outputs = [dense_2])

## === cell 12
def SpearmanCorrCoeff_2(A, B):
  overall_score = 0
  x1 = np.random.normal(loc = 1e-8, scale = 1e-12, size = A.shape[0])
  x2 = np.random.normal(loc = 1e-8, scale = 1e-12, size = B.shape[0])
  for index, col in enumerate(columns):
      overall_score += spearmanr(A[:, index]+x1, B[:, index]).correlation/30
  return overall_score

def tf_SpearmanCorrCoeff(A, B):
  result = tf.numpy_function(SpearmanCorrCoeff_2, [A, B], Tout = tf.double)
  return result

## === cell 13
model_2.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),loss=['binary_crossentropy'], metrics = [tf_SpearmanCorrCoeff])
model_2.fit([tf.convert_to_tensor(train_encoded_inputs_tensor)], tf.convert_to_tensor(labels.values), batch_size = 16, epochs = 20)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/287779194.py in <cell line: 0>()
      1 model_2.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),loss=['binary_crossentropy'], metrics = [tf_SpearmanCorrCoeff])
----> 2 model_2.fit([tf.convert_to_tensor(train_encoded_inputs_tensor)], tf.convert_to_tensor(labels.values), batch_size = 16, epochs = 20)

NameError: name 'train_encoded_inputs_tensor' is not defined

## === cell 14
ans = model_2.predict([tf.convert_to_tensor(test_encoded_inputs_tensor)])

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3603075391.py in <cell line: 0>()
----> 1 ans = model_2.predict([tf.convert_to_tensor(test_encoded_inputs_tensor)])

NameError: name 'test_encoded_inputs_tensor' is not defined

## === cell 15
sample_submission = pd.read_csv(DIR+'sample_submission.csv')

## === cell 16
a = pd.DataFrame(ans, columns = sample_submission.columns[1:])
a['qa_id'] = sample_submission.qa_id
cols = a.columns.tolist()
a = a[cols[-1:] + cols[:-1]]

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3228299396.py in <cell line: 0>()
----> 1 a = pd.DataFrame(ans, columns = sample_submission.columns[1:])
      2 a['qa_id'] = sample_submission.qa_id
      3 cols = a.columns.tolist()
      4 a = a[cols[-1:] + cols[:-1]]

NameError: name 'ans' is not defined

## === cell 17
a.to_csv('submission.csv', index = False)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3675781894.py in <cell line: 0>()
----> 1 a.to_csv('submission.csv', index = False)

NameError: name 'a' is not defined
