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
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LinearRegression

device = "cuda" if torch.cuda.is_available() else "cpu"



## === cell 1
model_path = "/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM/snapshots/fdfdbc22d7b15b174d2375ec022e32c7a3539276"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, local_files_only=True
)
model.to(device)
model.eval()




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
/tmp/ipykernel_55/608798149.py in <cell line: 0>()
      1 model_path = "/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM/snapshots/fdfdbc22d7b15b174d2375ec022e32c7a3539276"
----> 2 tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
      3 model = AutoModelForSequenceClassification.from_pretrained(
      4     model_path, local_files_only=True
      5 )

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
def inference(input_text):
    encoded = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    preds = outputs.logits.squeeze().cpu().numpy()
    return preds  # shape (6,)




## === cell 3
df_train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)



## === cell 4
y_true_train = []
y_pred_train = []

for row in tqdm(df_train.itertuples(), total=len(df_train)):
    pred = inference(row.full_text)
    y_pred_train.append(pred)
    y_true_train.append(row.score)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/239896997.py in <cell line: 0>()
      3 
      4 for row in tqdm(df_train.itertuples(), total=len(df_train)):
----> 5     pred = inference(row.full_text)
      6     y_pred_train.append(pred)
      7     y_true_train.append(row.score)

/tmp/ipykernel_55/3367555888.py in inference(input_text)
      1 def inference(input_text):
      2     # Tokenize and move tensors to the correct device
----> 3     encoded = tokenizer(
      4         input_text,
      5         return_tensors="pt",

NameError: name 'tokenizer' is not defined

## === cell 5
X = np.vstack(y_pred_train)  # shape (n_samples, 6)
y = np.array(y_true_train)  # shape (n_samples,)

lr = LinearRegression()
lr.fit(X, y)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3426704108.py in <cell line: 0>()
      1 # Convert lists to numpy arrays with correct dimensions
----> 2 X = np.vstack(y_pred_train)  # shape (n_samples, 6)
      3 y = np.array(y_true_train)  # shape (n_samples,)
      4 
      5 lr = LinearRegression()

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate

## === cell 6
print("coefficients shape:", lr.coef_.shape)
print("intercept:", lr.intercept_)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1806707590.py in <cell line: 0>()
----> 1 print("coefficients shape:", lr.coef_.shape)
      2 print("intercept:", lr.intercept_)
      3 

NameError: name 'lr' is not defined

## === cell 7
df_pred = pd.DataFrame(
    y_pred_train,
    columns=[
        "cohesion",
        "syntax",
        "vocabulary",
        "phraseology",
        "grammar",
        "conventions",
    ],
)
df_pred["predict_1to6"] = np.clip(lr.predict(X), 1, 6).round()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3290846368.py in <cell line: 0>()
     11     ],
     12 )
---> 13 df_pred["predict_1to6"] = np.clip(lr.predict(X), 1, 6).round()
     14 

NameError: name 'lr' is not defined

## === cell 8
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)



## === cell 9
y_pred_test = []
for row in tqdm(df_test.itertuples(), total=len(df_test)):
    pred = inference(row.full_text)
    y_pred_test.append(pred)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/905106193.py in <cell line: 0>()
      1 y_pred_test = []
      2 for row in tqdm(df_test.itertuples(), total=len(df_test)):
----> 3     pred = inference(row.full_text)
      4     y_pred_test.append(pred)
      5 

/tmp/ipykernel_55/3367555888.py in inference(input_text)
      1 def inference(input_text):
      2     # Tokenize and move tensors to the correct device
----> 3     encoded = tokenizer(
      4         input_text,
      5         return_tensors="pt",

NameError: name 'tokenizer' is not defined

## === cell 10
X_test = np.vstack(y_pred_test)
y_pred_regress_test = np.clip(lr.predict(X_test), 1, 6).round().astype(int)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4166345691.py in <cell line: 0>()
----> 1 X_test = np.vstack(y_pred_test)
      2 y_pred_regress_test = np.clip(lr.predict(X_test), 1, 6).round().astype(int)
      3 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate

## === cell 11
submission = pd.DataFrame(
    {"essay_id": df_test["essay_id"], "score": y_pred_regress_test}
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3567651215.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"essay_id": df_test["essay_id"], "score": y_pred_regress_test}
      3 )
      4 

NameError: name 'y_pred_regress_test' is not defined

## === cell 12
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3990991418.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined
