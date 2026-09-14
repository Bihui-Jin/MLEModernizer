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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.3173781628771867

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
DIR = '/kaggle/input/google-quest-challenge'
BATCH_SIZE = 6

## === cell 2
import pandas as pd
import numpy as np
import transformers
import tensorflow as tf
from scipy.stats import spearmanr
import tensorflow_hub as hub
import re
from sklearn.preprocessing import OneHotEncoder
import gc
from sklearn.model_selection import GroupKFold,KFold
from scipy.stats import spearmanr

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
transformer = transformers.TFDistilBertModel.from_pretrained('/kaggle/input/distil-bert-model/model_distil')
tokenizer = transformers.DistilBertTokenizer.from_pretrained('/kaggle/input/distil-bert-model/model_distil')


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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/distil-bert-model/model_distil'. Use `repo_type` argument if needed.

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/distil-bert-model/model_distil'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/1535661682.py in <cell line: 0>()
      5 # encoder.save_pretrained('/kaggle/working/models')
      6 # tokenizer.save_pretrained('/kaggle/working/models')
----> 7 transformer = transformers.TFDistilBertModel.from_pretrained('/kaggle/input/distil-bert-model/model_distil')
      8 tokenizer = transformers.DistilBertTokenizer.from_pretrained('/kaggle/input/distil-bert-model/model_distil')
      9 #embed = hub.load('/kaggle/input/universal-sentence-encoder')

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

OSError: Can't load the configuration of '/kaggle/input/distil-bert-model/model_distil'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/kaggle/input/distil-bert-model/model_distil' is the correct path to a directory containing a config.json file

## === cell 5
train_df = pd.read_csv(DIR+'/train.csv')
test_df = pd.read_csv(DIR+'/test.csv')

## === cell 6
def get_encoder_embed(string_list):
  batch_size = 4
  token_batches = []
  max_len = 512
  mask = []
  embed = []
  n = len(string_list)
  for i in range(0, n, batch_size):
    pad = []
    tokens = tokenizer.batch_encode_plus(string_list[i:i+batch_size], max_length=max_len, truncation=True)['input_ids']
    for index, _ in enumerate(tokens):
      x = np.array(tokens[index] + [0]*(max_len - len(tokens[index])))
      tokens[index] = x.tolist()
        
      pad.append(np.where(x == 0, 0, 1).tolist())
    mask.append(pad)
    token_batches.append(tokens)
    embed.append(transformer({'input_ids':np.array(token_batches[-1], dtype = np.int32), 'attention_mask':np.array(mask[-1])})[0][:, 0, :])
  return (token_batches, mask, tf.concat(embed, axis = 0))#transformer({'input_ids':token_batches, 'attention_mask':mask})) 



## === cell 7
question_ids, question_masks = {}, {}
answer_ids, answer_masks = {}, {}
question_encode = {}
answer_encode = {}

question_ids['train'], question_masks['train'], question_encode['train'] = get_encoder_embed(train_df.question_body.tolist())
question_ids['test'], question_masks['test'], question_encode['test'] = get_encoder_embed(test_df.question_body.tolist())

answer_ids['train'], answer_masks['train'], answer_encode['train'] = get_encoder_embed(train_df.answer.tolist())
answer_ids['test'], answer_masks['test'], answer_encode['test'] = get_encoder_embed(test_df.answer.tolist())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3371379306.py in <cell line: 0>()
      4 answer_encode = {}
      5 
----> 6 question_ids['train'], question_masks['train'], question_encode['train'] = get_encoder_embed(train_df.question_body.tolist())
      7 question_ids['test'], question_masks['test'], question_encode['test'] = get_encoder_embed(test_df.question_body.tolist())
      8 

/tmp/ipykernel_11/3332475907.py in get_encoder_embed(string_list)
      8   for i in range(0, n, batch_size):
      9     pad = []
---> 10     tokens = tokenizer.batch_encode_plus(string_list[i:i+batch_size], max_length=max_len, truncation=True)['input_ids']
     11     for index, _ in enumerate(tokens):
     12       x = np.array(tokens[index] + [0]*(max_len - len(tokens[index])))

NameError: name 'tokenizer' is not defined

## === cell 8
module_url = '/kaggle/input/universal-sentence-encoder'
model = hub.load(module_url)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2102462302.py in <cell line: 0>()
      1 module_url = '/kaggle/input/universal-sentence-encoder'
----> 2 model = hub.load(module_url)

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/module_v2.py in load(handle, tags, options)
     98   if not isinstance(handle, str):
     99     raise ValueError("Expected a string, got %s" % handle)
--> 100   module_path = resolve(handle)
    101   is_hub_module_v1 = tf.io.gfile.exists(_get_module_proto_path(module_path))
    102   if tags is None and is_hub_module_v1:

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/module_v2.py in resolve(handle)
     53     A string representing the Module path.
     54   """
---> 55   return registry.resolver(handle)
     56 
     57 

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/registry.py in __call__(self, *args, **kwargs)
     47     for impl in reversed(self._impls):
     48       if impl.is_supported(*args, **kwargs):
---> 49         return impl(*args, **kwargs)
     50       else:
     51         fails.append(type(impl).__name__)

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/resolver.py in __call__(self, handle)
    497   def __call__(self, handle):
    498     if not tf.compat.v1.gfile.Exists(handle):
--> 499       raise IOError("%s does not exist." % handle)
    500     return handle
    501 

OSError: /kaggle/input/universal-sentence-encoder does not exist.

## === cell 9
train_df['netloc'] = train_df.url.apply(lambda x:re.search(r'//.*?\.', x).group(0)[2:-1])
test_df['netloc'] = test_df.url.apply(lambda x:re.search(r'//.*?\.', x).group(0)[2:-1])
ohe = OneHotEncoder()
features = ['netloc', 'category']
merged = pd.concat([train_df[features], test_df[features]])
ohe.fit(merged)
features_train = ohe.transform(train_df[features]).toarray()
features_test = ohe.transform(test_df[features]).toarray()

## === cell 10
def get_universal_encoder(df):
  cols = ['question_title', 'question_body', 'answer']
  universal_embed = {}
  for col in cols:
    x = df[col].str.replace('?', '.').replace('!', '.').tolist()
    batch_size = 4
    curr_embed = []
    for i in range(0, len(x), batch_size):
      curr_embed.append(model(x[i:i+4]).numpy())
    universal_embed[col] = curr_embed
  for col in cols:
    universal_embed[col] = np.vstack(universal_embed[col])
  return universal_embed


## === cell 11
train_universal_embed = get_universal_encoder(train_df)
test_universal_embed = get_universal_encoder(test_df)
tf.keras.backend.clear_session()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1905316420.py in <cell line: 0>()
----> 1 train_universal_embed = get_universal_encoder(train_df)
      2 test_universal_embed = get_universal_encoder(test_df)
      3 tf.keras.backend.clear_session()

/tmp/ipykernel_11/1124174262.py in get_universal_encoder(df)
      9     curr_embed = []
     10     for i in range(0, len(x), batch_size):
---> 11       curr_embed.append(model(x[i:i+4]).numpy())
     12     universal_embed[col] = curr_embed
     13   for col in cols:

NameError: name 'model' is not defined

## === cell 12
l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)

cos_dist = lambda x, y: (x*y).sum(axis=1)

dist_features_train = np.array([
    l2_dist(train_universal_embed['question_title'], train_universal_embed['answer']),
    l2_dist(train_universal_embed['question_body'], train_universal_embed['answer']),
    l2_dist(train_universal_embed['question_body'], train_universal_embed['question_title']),
    cos_dist(train_universal_embed['question_title'], train_universal_embed['answer']),
    cos_dist(train_universal_embed['question_body'], train_universal_embed['answer']),
    cos_dist(train_universal_embed['question_body'], train_universal_embed['question_title'])
]).T

dist_features_test = np.array([
    l2_dist(test_universal_embed['question_title'], test_universal_embed['answer']),
    l2_dist(test_universal_embed['question_body'], test_universal_embed['answer']),
    l2_dist(test_universal_embed['question_body'], test_universal_embed['question_title']),
    cos_dist(test_universal_embed['question_title'], test_universal_embed['answer']),
    cos_dist(test_universal_embed['question_body'], test_universal_embed['answer']),
    cos_dist(test_universal_embed['question_body'], test_universal_embed['question_title'])
]).T

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3560027135.py in <cell line: 0>()
      4 
      5 dist_features_train = np.array([
----> 6     l2_dist(train_universal_embed['question_title'], train_universal_embed['answer']),
      7     l2_dist(train_universal_embed['question_body'], train_universal_embed['answer']),
      8     l2_dist(train_universal_embed['question_body'], train_universal_embed['question_title']),

NameError: name 'train_universal_embed' is not defined

## === cell 13
X_train = np.hstack([question_encode['train'], answer_encode['train'], dist_features_train, features_train] + \
[value for _, value in train_universal_embed.items()])

X_test = np.hstack([question_encode['test'], answer_encode['test'], dist_features_test, features_test] + \
[value for _, value in test_universal_embed.items()])

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4101454108.py in <cell line: 0>()
----> 1 X_train = np.hstack([question_encode['train'], answer_encode['train'], dist_features_train, features_train] + \
      2 [value for _, value in train_universal_embed.items()])
      3 
      4 X_test = np.hstack([question_encode['test'], answer_encode['test'], dist_features_test, features_test] + \
      5 [value for _, value in test_universal_embed.items()])

KeyError: 'train'

## === cell 14
class SpearmanRhoCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, validation_data, patience):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        rho_val = 0
        for ind in range(self.y_val.shape[1]):
          rho_val += spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation
        rho_val /= self.y_val.shape[1]
        if rho_val >= self.value:
            self.value = rho_val
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True
        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')
        return rho_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

## === cell 15
def create_model():
  input = tf.keras.Input(shape=(X_train.shape[1],))
  x = tf.keras.layers.Dense(128, activation='relu')(input)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid')(x)
  model = tf.keras.Model(inputs=input, outputs=x)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=['binary_crossentropy'], )#metrics = [tf_SpearmanCorrCoeff])

  return model

## === cell 16
Y_train = train_df.iloc[:, 11:-1].values

## === cell 17
init_lr = 2e-4
def scheduler(epoch, _):
  if epoch < 2:
    return init_lr
  else:
    if epoch < 20:
      return init_lr*np.exp(-epoch/20)
    else:
      return init_lr*np.exp(-20/20)
    
lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

## === cell 18
kf = KFold(n_splits=5, random_state = 10, shuffle=True).split(X=X_train)
valid_preds = []
for fold, (train_idx, valid_idx) in enumerate(kf):
  tf.keras.backend.clear_session()
  model = create_model()
  model.fit(X_train[train_idx], Y_train[train_idx], validation_data = (X_train[valid_idx], Y_train[valid_idx]), epochs=100, batch_size=64,verbose = 0,
  callbacks = [SpearmanRhoCallback(training_data=(X_train[train_idx], Y_train[train_idx]), validation_data=(X_train[valid_idx], Y_train[valid_idx]), patience=5),
               lr_schedule])
  print("##########################################################")


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1674027601.py in <cell line: 0>()
----> 1 kf = KFold(n_splits=5, random_state = 10, shuffle=True).split(X=X_train)
      2 valid_preds = []
      3 for fold, (train_idx, valid_idx) in enumerate(kf):
      4   tf.keras.backend.clear_session()
      5   model = create_model()

NameError: name 'X_train' is not defined

## === cell 19
ans = model.predict(X_test)
sample_submission = pd.read_csv(DIR+'/sample_submission.csv')
sample_submission.iloc[:, 1:] = ans

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1574495261.py in <cell line: 0>()
----> 1 ans = model.predict(X_test)
      2 sample_submission = pd.read_csv(DIR+'/sample_submission.csv')
      3 sample_submission.iloc[:, 1:] = ans

NameError: name 'model' is not defined

## === cell 20
sample_submission.to_csv('submission.csv', index = False)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1882623384.py in <cell line: 0>()
----> 1 sample_submission.to_csv('submission.csv', index = False)

NameError: name 'sample_submission' is not defined

## === cell 21
sample_submission.head()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3668618413.py in <cell line: 0>()
----> 1 sample_submission.head()

NameError: name 'sample_submission' is not defined
