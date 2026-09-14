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

0.3655933516945904

# 6. Current score

-0.01016

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -0.01016) has done: 'I fix the immediate runtime blockers by removing the broken `transformers` imports (they’re not usable in this environment) and by switching tokenization/model loading to a lightweight, local TF‑IDF + Ridge baseline that runs with the preinstalled sklearn. This keeps the same overall pipeline semantics (text in → 30 continuous targets in [0,1] → submission.csv) while ensuring an end‑to‑end run that actually writes a valid CSV. I also ensure the submission columns and row order exactly match `sample_submission.csv` and clip predictions into [0,1] as required. Finally, I keep the rest of the notebook structure intact but bypass the dead BERT/data-cache cells that currently prevent producing any submission.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline



## === cell 2
sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
train = pd.read_csv("../input/google-quest-challenge/train.csv")



## === cell 3
train.columns.values



## === cell 4
output_columns = train.columns.values[11:]
input_columns = train.columns.values[[1, 2, 5]]



## === cell 5
question_output_columns = [col for col in output_columns if "question" in col]
answer_output_colmns = [
    col for col in output_columns if col not in question_output_columns
]



## === cell 6
len(question_output_columns)



## === cell 7
tokenizer = None



## === cell 8
max_length_map = {"question_title": 32, "question_body": 512, "answer": 512}




## === cell 9
def txt_re(txt):
    if txt is None or (isinstance(txt, float) and np.isnan(txt)):
        return ""
    txt = str(txt).strip()
    txt = re.sub(r"https?.*$", "", txt)
    txt = re.sub(r"https?.*\s", "", txt)
    txt = re.sub(r"\n+", " ", txt)
    txt = re.sub(r"\r+", " ", txt)
    txt = re.sub(r"\t+", " ", txt)
    txt = re.sub(r"&gt;", ">", txt)
    txt = re.sub(r"&lt;", "<", txt)
    txt = re.sub(r"&amp;", "&", txt)
    txt = re.sub(r"&quot;", '"', txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt




## === cell 10
def get_input(txt, pair_txt, tokenizer, max_length):
    raise RuntimeError(
        "BERT tokenization disabled in this environment; using TF-IDF baseline instead."
    )




## === cell 11
def computer_input_array(df):
    texts = (
        df["question_title"].map(txt_re).fillna("")
        + " [SEP] "
        + df["question_body"].map(txt_re).fillna("")
        + " [SEP] "
        + df["answer"].map(txt_re).fillna("")
    ).tolist()
    return texts, None




## === cell 12
question_train, answer_train = computer_input_array(train)
question_test, answer_test = computer_input_array(test)



## === cell 13
answer_test



## === cell 14
labels = train[output_columns].values.astype(np.float32)



## === cell 15
train_dict = {"question": question_train, "answer": answer_train, "label": labels}
test_dict = {"question": question_test, "answer": answer_test}



## === cell 16
_ = None



## === cell 17
import os

os.makedirs("./data", exist_ok=True)



## === cell 18
os.makedirs("./model", exist_ok=True)



## === cell 19
import torch

torch.save(train_dict, "./data/train_data.t7")
torch.save(test_dict, "./data/test_data.t7")



## === cell 20
data = torch.load("./data/train_data.t7")
test_data = torch.load("./data/test_data.t7")



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
UnpicklingError                           Traceback (most recent call last)
/tmp/ipykernel_55/1167096116.py in <cell line: 0>()
      1 # Added missing cell index (original jumped from 20 to 22).
      2 # Score-neutral: load saved data to match original structure.
----> 3 data = torch.load("./data/train_data.t7")
      4 test_data = torch.load("./data/test_data.t7")
      5 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1468                         )
   1469                     except pickle.UnpicklingError as e:
-> 1470                         raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
   1471                 return _load(
   1472                     opened_zipfile,

UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct was not an allowed global by default. Please use `torch.serialization.add_safe_globals([_reconstruct])` or the `torch.serialization.safe_globals([_reconstruct])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.

## === cell 21
import numpy as np
from scipy.stats import spearmanr



## === cell 22
import torch.nn as nn


class Model_v1(nn.Module):
    def __init__(self):
        super().__init__()
        raise RuntimeError(
            "BERT model disabled in this environment; using TF-IDF baseline instead."
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError("Disabled.")




## === cell 23
class Model_v2(nn.Module):
    def __init__(self):
        super().__init__()
        raise RuntimeError(
            "BERT model disabled in this environment; using TF-IDF baseline instead."
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError("Disabled.")




## === cell 24
X_train_text = np.array(data["question"], dtype=object)
X_test_text = np.array(test_data["question"], dtype=object)
y_train = np.array(data["label"], dtype=np.float32)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4118195110.py in <cell line: 0>()
      2 # Prepare sklearn model pipeline (TF-IDF + Ridge) for multi-target regression.
      3 # This is a standard baseline for this competition and should achieve a non-trivial score.
----> 4 X_train_text = np.array(data["question"], dtype=object)
      5 X_test_text = np.array(test_data["question"], dtype=object)
      6 y_train = np.array(data["label"], dtype=np.float32)

NameError: name 'data' is not defined

## === cell 25
tfidf_ridge = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                strip_accents="unicode",
                lowercase=True,
                sublinear_tf=True,
            ),
        ),
        ("reg", MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))),
    ]
)



## === cell 26
tfidf_ridge.fit(X_train_text, y_train)
pred = tfidf_ridge.predict(X_test_text).astype(np.float32)
pred = np.clip(pred, 0.0, 1.0)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3972390748.py in <cell line: 0>()
      1 # Train and predict
----> 2 tfidf_ridge.fit(X_train_text, y_train)
      3 pred = tfidf_ridge.predict(X_test_text).astype(np.float32)
      4 pred = np.clip(pred, 0.0, 1.0)
      5 

NameError: name 'X_train_text' is not defined

## === cell 27
assert pred.shape[0] == len(test), (pred.shape, len(test))
assert pred.shape[1] == len(output_columns), (pred.shape, len(output_columns))



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/249073984.py in <cell line: 0>()
      1 # Ensure prediction shape matches (n_test, 30)
----> 2 assert pred.shape[0] == len(test), (pred.shape, len(test))
      3 assert pred.shape[1] == len(output_columns), (pred.shape, len(output_columns))
      4 

NameError: name 'pred' is not defined

## === cell 28
submission = sample_submission.copy()
target_cols = [c for c in submission.columns if c != "qa_id"]

missing = [c for c in target_cols if c not in output_columns]
if missing:
    raise ValueError(f"Targets missing from training columns: {missing}")

col_idx = [list(output_columns).index(c) for c in target_cols]
submission[target_cols] = pred[:, col_idx]

if not np.array_equal(submission["qa_id"].values, test["qa_id"].values):
    tmp = pd.DataFrame(pred[:, col_idx], columns=target_cols)
    tmp["qa_id"] = test["qa_id"].values
    submission = sample_submission[["qa_id"]].merge(tmp, on="qa_id", how="left")
    submission[target_cols] = submission[target_cols].fillna(0.5)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1254269404.py in <cell line: 0>()
     10 # Reorder predictions to match sample submission columns
     11 col_idx = [list(output_columns).index(c) for c in target_cols]
---> 12 submission[target_cols] = pred[:, col_idx]
     13 
     14 # Ensure qa_id alignment: use test.qa_id order as in sample_submission (should match competition files),

NameError: name 'pred' is not defined

## === cell 29
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())




## === cell 30
def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)




## === cell 31
"""
Original BERT training loop removed/disabled due to environment constraints.
"""



## === cell 32
_ = None



## === cell 33
models = []



## === cell 34
len(models)



## === cell 35
final_predicts = []



## === cell 36
pres = None
test_output = None



## === cell 37
output_cols = [c for c in sample_submission.columns if c != "qa_id"]



## === cell 38
output = submission



## === cell 39
order = ["qa_id"] + output_cols



## === cell 40
output = output[order]



## === cell 41
output.head()



## === cell 42
output.to_csv("submission.csv", index=False)
print("Final submission saved to submission.csv")
