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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.3194937486952924

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

from tqdm import tqdm

from sklearn.linear_model import LinearRegression

## === cell 1
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM/snapshots/fdfdbc22d7b15b174d2375ec022e32c7a3539276")
model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM/snapshots/fdfdbc22d7b15b174d2375ec022e32c7a3539276")

## --- ERROR in cell 1, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM/snapshots/fdfdbc22d7b15b174d2375ec022e32c7a3539276'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/2842573055.py in <cell line: 0>()
----> 1 tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM/snapshots/fdfdbc22d7b15b174d2375ec022e32c7a3539276")
      2 model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM/snapshots/fdfdbc22d7b15b174d2375ec022e32c7a3539276")

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
    981 
    982         # Next, let's try to use the tokenizer_config file to get the tokenizer class.
--> 983         tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
    984         if "_commit_hash" in tokenizer_config:
    985             kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in get_tokenizer_config(pretrained_model_name_or_path, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, **kwargs)
    813 
    814     commit_hash = kwargs.get("_commit_hash", None)
--> 815     resolved_config_file = cached_file(
    816         pretrained_model_name_or_path,
    817         TOKENIZER_CONFIG_FILE,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM/snapshots/fdfdbc22d7b15b174d2375ec022e32c7a3539276'. Use `repo_type` argument if needed.

## === cell 2
model.to(device)
model.eval()

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3622578787.py in <cell line: 0>()
----> 1 model.to(device)
      2 model.eval()

NameError: name 'model' is not defined

## === cell 3
def inference(input_text):
    encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    encoded_input.to(device)

    with torch.no_grad():
        outputs = model(**encoded_input)

    predictions = outputs.logits.squeeze()

    predicted_scores = predictions.cpu().numpy()  
    item_names = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

    return predicted_scores





## === cell 4
df_train = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')

## === cell 5
y_true_train = []
y_pred_train = []

for t in tqdm(df_train.itertuples()):
    predict = inference(t.full_text)
    label = t.score    
    y_true_train.append(label)
    y_pred_train.append(predict)
    torch.cuda.empty_cache()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3339011355.py in <cell line: 0>()
      3 
      4 for t in tqdm(df_train.itertuples()):
----> 5     predict = inference(t.full_text)
      6     label = t.score
      7     y_true_train.append(label)

/tmp/ipykernel_11/2762151399.py in inference(input_text)
      1 def inference(input_text):
----> 2     encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=64)
      3     encoded_input.to(device)
      4 
      5     # Perform the prediction

NameError: name 'tokenizer' is not defined

## === cell 6
lr = LinearRegression()

X = y_pred_train
Y = y_true_train

lr.fit(X, Y)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/926135415.py in <cell line: 0>()
      4 Y = y_true_train
      5 
----> 6 lr.fit(X, Y)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in fit(self, X, y, sample_weight)
    646         accept_sparse = False if self.positive else ["csr", "csc", "coo"]
    647 
--> 648         X, y = self._validate_data(
    649             X, y, accept_sparse=accept_sparse, y_numeric=True, multi_output=True
    650         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    900             # If input is 1D raise error
    901             if array.ndim == 1:
--> 902                 raise ValueError(
    903                     "Expected 2D array, got 1D array instead:\narray={}.\n"
    904                     "Reshape your data either using array.reshape(-1, 1) if "

ValueError: Expected 2D array, got 1D array instead:
array=[].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

## === cell 7
print('coefficient = ', lr.coef_)
print('intercept = ', lr.intercept_)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2584904912.py in <cell line: 0>()
----> 1 print('coefficient = ', lr.coef_)
      2 print('intercept = ', lr.intercept_)

AttributeError: 'LinearRegression' object has no attribute 'coef_'

## === cell 8
df_pred = pd.DataFrame(y_pred_train, columns=["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"])
df_pred['predict_1to6'] = np.clip(lr.predict(X), 1, 6).round()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/815450897.py in <cell line: 0>()
      1 df_pred = pd.DataFrame(y_pred_train, columns=["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"])
----> 2 df_pred['predict_1to6'] = np.clip(lr.predict(X), 1, 6).round()

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    333 
    334     def _decision_function(self, X):
--> 335         check_is_fitted(self)
    336 
    337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 9
df_test = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv')

## === cell 10
y_pred_test = []

for t in tqdm(df_test.itertuples()):
    predict = inference(t.full_text)
    y_pred_test.append(predict)
    torch.cuda.empty_cache()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4215918276.py in <cell line: 0>()
      2 
      3 for t in tqdm(df_test.itertuples()):
----> 4     predict = inference(t.full_text)
      5     y_pred_test.append(predict)
      6     torch.cuda.empty_cache()

/tmp/ipykernel_11/2762151399.py in inference(input_text)
      1 def inference(input_text):
----> 2     encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=64)
      3     encoded_input.to(device)
      4 
      5     # Perform the prediction

NameError: name 'tokenizer' is not defined

## === cell 11
X_test = y_pred_test
y_pred_regress_test = np.clip(lr.predict(X_test), 1, 6).round().astype('int8')

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2852841301.py in <cell line: 0>()
      1 X_test = y_pred_test
----> 2 y_pred_regress_test = np.clip(lr.predict(X_test), 1, 6).round().astype('int8')

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    333 
    334     def _decision_function(self, X):
--> 335         check_is_fitted(self)
    336 
    337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 12
pd.DataFrame({'essay_id':df_test['essay_id'], 'score':y_pred_regress_test})

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2094989695.py in <cell line: 0>()
----> 1 pd.DataFrame({'essay_id':df_test['essay_id'], 'score':y_pred_regress_test})

NameError: name 'y_pred_regress_test' is not defined

## === cell 13
pd.DataFrame({'essay_id':df_test['essay_id'], 'score':y_pred_regress_test}).to_csv('submission.csv', index=False)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3205442701.py in <cell line: 0>()
----> 1 pd.DataFrame({'essay_id':df_test['essay_id'], 'score':y_pred_regress_test}).to_csv('submission.csv', index=False)

NameError: name 'y_pred_regress_test' is not defined
