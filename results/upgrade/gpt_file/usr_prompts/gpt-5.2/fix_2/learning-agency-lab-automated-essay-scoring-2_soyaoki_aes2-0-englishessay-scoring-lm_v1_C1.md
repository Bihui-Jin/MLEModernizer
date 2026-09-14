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
import os
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LinearRegression

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



## === cell 1
MODEL_DIR = (
    "/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM"
)

snapshots_dir = os.path.join(MODEL_DIR, "snapshots")
if os.path.isdir(snapshots_dir):
    snapshot_subdirs = [
        os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir)
    ]
    snapshot_subdirs = [d for d in snapshot_subdirs if os.path.isdir(d)]
    snapshot_subdirs.sort()
    if len(snapshot_subdirs) > 0:
        MODEL_LOAD_PATH = snapshot_subdirs[-1]
    else:
        MODEL_LOAD_PATH = MODEL_DIR
else:
    MODEL_LOAD_PATH = MODEL_DIR

print("MODEL_LOAD_PATH:", MODEL_LOAD_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_LOAD_PATH, local_files_only=True
)



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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_55/2785496616.py in <cell line: 0>()
     22 print("MODEL_LOAD_PATH:", MODEL_LOAD_PATH)
     23 
---> 24 tokenizer = AutoTokenizer.from_pretrained(MODEL_LOAD_PATH, local_files_only=True)
     25 model = AutoModelForSequenceClassification.from_pretrained(
     26     MODEL_LOAD_PATH, local_files_only=True

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/englishessay-scoring-lm/models--rong4ivy--EnglishEssay_Scoring_LM'. Use `repo_type` argument if needed.

## === cell 2
model.to(device)
model.eval()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/646965245.py in <cell line: 0>()
----> 1 model.to(device)
      2 model.eval()
      3 
      4 

NameError: name 'model' is not defined

## === cell 3
@torch.inference_mode()
def inference_batch(text_list, max_length=256, batch_size=32):
    feats = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = out.logits

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.shape[1] == 1:
            logits = logits.repeat(1, 6)

        feats.append(logits.detach().float().cpu().numpy())
    return np.vstack(feats)




## === cell 4
df_train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)

print(df_train.shape, df_test.shape)
print(df_train.columns.tolist())



## === cell 5
MAX_TRAIN_ROWS = (
    12000  # chosen to be safe under the 600s timeout in Kaggle CPU/GPU environments
)

if len(df_train) > MAX_TRAIN_ROWS:
    df_train_sub = df_train.sample(n=MAX_TRAIN_ROWS, random_state=SEED).reset_index(
        drop=True
    )
else:
    df_train_sub = df_train.copy()

train_texts = df_train_sub["full_text"].astype(str).tolist()
y_true_train = df_train_sub["score"].astype(float).values

X_train = inference_batch(train_texts, max_length=256, batch_size=32)
print("X_train shape:", X_train.shape, "y_true_train shape:", y_true_train.shape)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/590475430.py in <cell line: 0>()
     16 y_true_train = df_train_sub["score"].astype(float).values
     17 
---> 18 X_train = inference_batch(train_texts, max_length=256, batch_size=32)
     19 print("X_train shape:", X_train.shape, "y_true_train shape:", y_true_train.shape)
     20 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/4246563062.py in inference_batch(text_list, max_length, batch_size)
      6     for i in range(0, len(text_list), batch_size):
      7         batch_texts = text_list[i : i + batch_size]
----> 8         enc = tokenizer(
      9             batch_texts,
     10             return_tensors="pt",

NameError: name 'tokenizer' is not defined

## === cell 6
lr = LinearRegression()
lr.fit(X_train, y_true_train)

print("coefficient =", lr.coef_)
print("intercept =", lr.intercept_)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1473995936.py in <cell line: 0>()
      1 lr = LinearRegression()
----> 2 lr.fit(X_train, y_true_train)
      3 
      4 print("coefficient =", lr.coef_)
      5 print("intercept =", lr.intercept_)

NameError: name 'X_train' is not defined

## === cell 7
y_pred_train_cont = lr.predict(X_train)
y_pred_train_1to6 = np.clip(y_pred_train_cont, 1, 6).round().astype("int8")
print("Train preds min/max:", y_pred_train_1to6.min(), y_pred_train_1to6.max())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1537621141.py in <cell line: 0>()
      1 # Train diagnostics (optional but quick)
----> 2 y_pred_train_cont = lr.predict(X_train)
      3 y_pred_train_1to6 = np.clip(y_pred_train_cont, 1, 6).round().astype("int8")
      4 print("Train preds min/max:", y_pred_train_1to6.min(), y_pred_train_1to6.max())
      5 

NameError: name 'X_train' is not defined

## === cell 8
test_texts = df_test["full_text"].astype(str).tolist()
X_test = inference_batch(test_texts, max_length=256, batch_size=32)
print("X_test shape:", X_test.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1406719938.py in <cell line: 0>()
      1 test_texts = df_test["full_text"].astype(str).tolist()
----> 2 X_test = inference_batch(test_texts, max_length=256, batch_size=32)
      3 print("X_test shape:", X_test.shape)
      4 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/4246563062.py in inference_batch(text_list, max_length, batch_size)
      6     for i in range(0, len(text_list), batch_size):
      7         batch_texts = text_list[i : i + batch_size]
----> 8         enc = tokenizer(
      9             batch_texts,
     10             return_tensors="pt",

NameError: name 'tokenizer' is not defined

## === cell 9
y_pred_regress_test = np.clip(lr.predict(X_test), 1, 6).round().astype("int8")

sub = pd.DataFrame(
    {"essay_id": df_test["essay_id"].values, "score": y_pred_regress_test}
)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2601299052.py in <cell line: 0>()
----> 1 y_pred_regress_test = np.clip(lr.predict(X_test), 1, 6).round().astype("int8")
      2 
      3 sub = pd.DataFrame(
      4     {"essay_id": df_test["essay_id"].values, "score": y_pred_regress_test}
      5 )

NameError: name 'X_test' is not defined
