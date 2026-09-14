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

0.3697054710738545

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge



## === cell 2
BASE_PATH = "/kaggle/input/google-quest-challenge"

sample_submission = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
test = pd.read_csv(f"{BASE_PATH}/test.csv")
train = pd.read_csv(f"{BASE_PATH}/train.csv")

print(train.shape, test.shape, sample_submission.shape)



## === cell 3
train.columns.values



## === cell 4
output_columns = train.columns.values[11:]
input_columns = train.columns.values[[1, 2, 5]]  # question_title, question_body, answer



## === cell 5
question_output_columns = [col for col in output_columns if "question" in col]
answer_output_colmns = [
    col for col in output_columns if col not in question_output_columns
]



## === cell 6
len(question_output_columns), len(answer_output_colmns), len(output_columns)



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
    return txt




## === cell 10
def get_input(txt, pair_txt, tokenizer, max_length):
    raise RuntimeError(
        "BERT tokenization disabled in this offline baseline. Use TF-IDF features instead."
    )




## === cell 11
def computer_input_array(df):
    raise RuntimeError(
        "BERT tensorization disabled in this offline baseline. Use TF-IDF features instead."
    )




## === cell 12
def build_text_frame(df: pd.DataFrame) -> pd.Series:
    title = df["question_title"].map(txt_re)
    body = df["question_body"].map(txt_re)
    ans = df["answer"].map(txt_re)
    return "title: " + title + " body: " + body + " answer: " + ans


X_train_text = build_text_frame(train)
X_test_text = build_text_frame(test)
y_train = train[list(output_columns)].astype(np.float32)

print(X_train_text.iloc[0][:200])
print(y_train.shape)



## === cell 13
print("Example test text length:", len(X_test_text.iloc[0]))



## === cell 14
labels = y_train.values.tolist()



## === cell 15
train_dict = {"text": X_train_text.values, "label": y_train.values}
test_dict = {"text": X_test_text.values}



## === cell 16
print("Working dir contents:", os.listdir("."))



## === cell 17
import os

os.makedirs("./data", exist_ok=True)



## === cell 18
os.makedirs("./model", exist_ok=True)



## === cell 19
import torch

torch.save(train_dict, "./data/train_data.t7")
torch.save(test_dict, "./data/test_data.t7")
print("Saved ./data/train_data.t7 and ./data/test_data.t7")



## === cell 20
pass



## === cell 21
import numpy as np
import torch
import torch.nn as nn




## === cell 22
class Model_v1(nn.Module):
    def __init__(self):
        super().__init__()
        raise RuntimeError("BERT model disabled in this offline baseline.")

    def forward(self, *args, **kwargs):
        raise RuntimeError("BERT model disabled in this offline baseline.")




## === cell 23
class Model_v2(nn.Module):
    def __init__(self):
        super().__init__()
        raise RuntimeError("BERT model disabled in this offline baseline.")

    def forward(self, *args, **kwargs):
        raise RuntimeError("BERT model disabled in this offline baseline.")




## === cell 24
pass



## === cell 25
from sklearn.model_selection import train_test_split



## === cell 26
data = torch.load("./data/train_data.t7")
test_data = torch.load("./data/test_data.t7")
print("Loaded saved data:", data.keys(), test_data.keys())



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
UnpicklingError                           Traceback (most recent call last)
/tmp/ipykernel_55/3421849160.py in <cell line: 0>()
      1 # Load back saved artifacts (optional) to mimic original flow; ensures files exist now.
----> 2 data = torch.load("./data/train_data.t7")
      3 test_data = torch.load("./data/test_data.t7")
      4 print("Loaded saved data:", data.keys(), test_data.keys())
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

## === cell 27
X_train = data["text"]
Y_train = data["label"]
X_test = test_data["text"]

print(
    "X_train shape:",
    X_train.shape,
    "Y_train shape:",
    Y_train.shape,
    "X_test shape:",
    X_test.shape,
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/382910702.py in <cell line: 0>()
      1 # Replace failing TensorDataset creation with sklearn-ready arrays.
----> 2 X_train = data["text"]
      3 Y_train = data["label"]
      4 X_test = test_data["text"]
      5 

NameError: name 'data' is not defined

## === cell 28
test_loader = None



## === cell 29
criterion = nn.BCELoss()




## === cell 30
def compute_spearmanr_ignore_nan(trues, preds):
    trues = np.asarray(trues)
    preds = np.asarray(preds)
    rhos = []
    for i in range(trues.shape[1]):
        t = pd.Series(trues[:, i]).rank(method="average")
        p = pd.Series(preds[:, i]).rank(method="average")
        rho = t.corr(p, method="pearson")
        rhos.append(rho)
    return float(np.nanmean(rhos))




## === cell 31
"""
(Original PyTorch/BERT training loop intentionally left disabled in this offline baseline.)
"""



## === cell 32
pass



## === cell 33
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
                max_features=200000,
            ),
        ),
        ("reg", MultiOutputRegressor(Ridge(alpha=1.0, random_state=42))),
    ]
)

print("Fitting TF-IDF + Ridge model...")
tfidf_ridge.fit(X_train, Y_train)
print("Fit complete.")



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3779575246.py in <cell line: 0>()
     20 
     21 print("Fitting TF-IDF + Ridge model...")
---> 22 tfidf_ridge.fit(X_train, Y_train)
     23 print("Fit complete.")
     24 

NameError: name 'X_train' is not defined

## === cell 34
models = [tfidf_ridge]
len(models)



## === cell 35
test_pred = tfidf_ridge.predict(X_test).astype(np.float32)
test_pred = np.clip(test_pred, 0.0, 1.0)
print(
    "Pred shape:",
    test_pred.shape,
    "min/max:",
    float(test_pred.min()),
    float(test_pred.max()),
)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1380286867.py in <cell line: 0>()
      1 # Predict on test.
----> 2 test_pred = tfidf_ridge.predict(X_test).astype(np.float32)
      3 # Ensure [0, 1] range as required by the competition.
      4 test_pred = np.clip(test_pred, 0.0, 1.0)
      5 print(

NameError: name 'X_test' is not defined

## === cell 36
test_output = test_pred.tolist()



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3954744894.py in <cell line: 0>()
      1 # BUGFIX: original averaging logic produced wrong shapes/types. Here we already have [n_rows, 30].
----> 2 test_output = test_pred.tolist()
      3 

NameError: name 'test_pred' is not defined

## === cell 37
output_cols = [c for c in sample_submission.columns if c != "qa_id"]
assert len(output_cols) == 30, f"Expected 30 target columns, got {len(output_cols)}"



## === cell 38
output = pd.DataFrame(test_output, columns=output_cols)
output.insert(0, "qa_id", test["qa_id"].values)

output = output[sample_submission.columns.tolist()]
print(output.head())



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3015772620.py in <cell line: 0>()
      1 # Build submission aligned with test qa_id.
----> 2 output = pd.DataFrame(test_output, columns=output_cols)
      3 output.insert(0, "qa_id", test["qa_id"].values)
      4 
      5 # Ensure exact column order matches sample_submission.

NameError: name 'test_output' is not defined

## === cell 39
order = sample_submission.columns.tolist()



## === cell 40
output = output[order]



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1610986860.py in <cell line: 0>()
----> 1 output = output[order]
      2 

NameError: name 'output' is not defined

## === cell 41
output.head()



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2010806425.py in <cell line: 0>()
----> 1 output.head()
      2 

NameError: name 'output' is not defined

## === cell 42
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3378144820.py in <cell line: 0>()
      1 # Write valid Kaggle submission.
----> 2 output.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", output.shape)

NameError: name 'output' is not defined
