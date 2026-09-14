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
Predicting the answers to questions in Hindi and Tamil.

## Metric
Word-level Jaccard score.

A Python implementation is provided below.

```
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric is:
\text{score} = \frac{1}{n} \sum_{i=1}^n \text{jaccard}(gt_i, dt_i)

where:
$n$ = number of documents

$\text{jaccard}$ = the function provided above

$gt_i$ = the ith ground truth

$dt_i$ = the ith prediction

## Submission Format
For each ID in the test set, you must predict the string that best answers the provided question based on the context. Note that the selected text needs to be quoted and complete to work correctly. Include punctuation, etc. The file should contain a header and have the following format:

```
id,PredictionString
8c8ee6504,"1"
3163c22d0,"2 string"
66aae423b,"4 word 6"
722085a7b,"1"
etc.
```

## Dataset 
**All files should be encoded as UTF-8.**

- **train.csv** - the training set, containing context, questions, and answers. Also includes the start character of the answer for disambiguation.
- **test.csv** - the test set, containing context and questions.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - a unique identifier
- `context` - the text of the Hindi/Tamil sample from which answers should be derived
- `question` - the question, in Hindi/Tamil
- `answer_text` (train only) - the answer to the question (manual annotation) (note: for test, this is what you are attempting to predict)
- `answer_start` (train only) - the starting character in `context` for the answer (determined using substring match during data preparation)
- `language` - whether the text in question is in Tamil or Hindi

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
rich==14.2.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
transformers==4.53.3
wordcloud==1.9.4

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        input/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        working/
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
```

-> data/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/chaii-hindi-and-tamil-question-answering/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/chaii-hindi-and-tamil-question-answering/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> data/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> input/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> (stopped after 10 files for performance)

# 5. Target score

0.4375553727149963

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import iplot

from wordcloud import WordCloud

from rich import print as _pprint




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def cprint(string):
    """
    Utility function for beautiful colored printing.
    """
    _pprint(f"[black]{string}[/black]")




## === cell 2
train_file = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/train.csv")
test_file = pd.read_csv("../input/chaii-hindi-and-tamil-question-answering/test.csv")
sample_sub = pd.read_csv(
    "../input/chaii-hindi-and-tamil-question-answering/sample_submission.csv"
)



## === cell 3
train_file.head()



## === cell 4
train_file.info()



## === cell 5
train_file.describe()



## === cell 6
test_file.head()



## === cell 7
test_file.info()



## === cell 8
test_file.describe()



## === cell 9
sample_sub.head()



## === cell 10
cprint("Total Training Examples: [green]{}[/green]".format(train_file.shape[0]))
cprint("Total Testing Examples: [green]{}[/green]".format(test_file.shape[0]))



## === cell 11
train_file["language"].value_counts()



## === cell 12
language_name = train_file["language"].value_counts().index.tolist()
language_val = train_file["language"].value_counts().tolist()

fig = px.bar(
    x=language_name,
    y=language_val,
    title="Training Samples by Language",
    labels={"x": "Language", "y": "Sample count"},
    color=language_val,
)
fig.show()



## === cell 13
fig = px.pie(
    names=language_name,
    values=language_val,
    title="Training Samples by Language - Pie Chart",
    color_discrete_sequence=px.colors.sequential.RdBu_r,
)
fig.show()



## === cell 14
hindi = train_file[train_file["language"] == "hindi"]["context"].str.len()
tamil = train_file[train_file["language"] == "tamil"]["context"].str.len()

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Histogram(x=list(hindi), name="Hindi Context"), row=1, col=1)
fig.add_trace(go.Histogram(x=list(tamil), name="Tamil Context"), row=1, col=2)

fig.update_layout(height=400, width=800, title_text="Character Count by Language")
iplot(fig)



## === cell 15
hindi = (
    train_file[train_file["language"] == "hindi"]["context"]
    .str.split()
    .map(lambda x: len(x))
)
tamil = (
    train_file[train_file["language"] == "tamil"]["context"]
    .str.split()
    .map(lambda x: len(x))
)

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Histogram(x=list(hindi), name="Hindi Context"), row=1, col=1)
fig.add_trace(go.Histogram(x=list(tamil), name="Tamil Context"), row=1, col=2)

fig.update_layout(
    height=400, width=800, title_text="Word Count Distribution by Language"
)
iplot(fig)



## === cell 16
hindi = (
    train_file[train_file["language"] == "hindi"]["context"]
    .str.split()
    .map(lambda x: [len(j) for j in x])
    .map(lambda x: np.mean(x))
    .to_list()
)
tamil = (
    train_file[train_file["language"] == "tamil"]["context"]
    .str.split()
    .map(lambda x: [len(j) for j in x])
    .map(lambda x: np.mean(x))
    .to_list()
)

fig = ff.create_distplot([hindi, tamil], ["Hindi", "Tamil"])
fig.update_layout(
    height=500, width=800, title_text="Average Word Length Distribution by Language"
)
iplot(fig)



## === cell 17
hindi = (
    train_file[train_file["language"] == "hindi"]["context"]
    .apply(lambda x: len(set(str(x).split())))
    .to_list()
)
tamil = (
    train_file[train_file["language"] == "tamil"]["context"]
    .apply(lambda x: len(set(str(x).split())))
    .to_list()
)

fig = ff.create_distplot([hindi, tamil], ["Hindi", "Tamil"])
fig.update_layout(
    height=500, width=800, title_text="Unique Word Count Distribution by Language"
)
iplot(fig)



## === cell 18
model_path = "../input/bbmcfs/bert-base-multilingual-cased-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=True)

qna = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)

predictions = []
for question, context in test_file[["question", "context"]].to_numpy():
    result = qna(context=context, question=question)
    predictions.append(result["answer"])



## --- ERROR in cell 18, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bbmcfs/bert-base-multilingual-cased-finetuned-squad'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_55/1806364861.py in <cell line: 0>()
      1 # Load the multilingual QA model from the local directory
      2 model_path = "../input/bbmcfs/bert-base-multilingual-cased-finetuned-squad"
----> 3 tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
      4 model = AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=True)
      5 

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bbmcfs/bert-base-multilingual-cased-finetuned-squad'. Use `repo_type` argument if needed.

## === cell 19
submission = pd.DataFrame()
submission["id"] = test_file["id"]
submission["PredictionString"] = predictions
submission.to_csv("submission.csv", index=False)

submission.head()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2304907211.py in <cell line: 0>()
      1 submission = pd.DataFrame()
      2 submission["id"] = test_file["id"]
----> 3 submission["PredictionString"] = predictions
      4 submission.to_csv("submission.csv", index=False)
      5 

NameError: name 'predictions' is not defined

## === cell 20
cprint("[red]Under Work! More stuff coming soon[/red] ⚠")
