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

3.13

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
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
text-unidecode==1.3
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.786623184549083

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
import re
from text_unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def preprocess_essay_text(text: str) -> str:
    text = resolve_encodings_and_normalize(text)
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r'\s+([?.!,"])', r"\1", text)
    text = re.sub(r",([^\s])", r", \1", text)
    return text


train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
train_df = pd.read_csv(train_path)
train_df["full_text"] = train_df["full_text"].apply(preprocess_essay_text)

tfidf_ridge = make_pipeline(
    TfidfVectorizer(max_features=60000, ngram_range=(1, 2)),
    Ridge(alpha=1.0, random_state=42),
)

X_train = train_df["full_text"].astype(str)
y_train = train_df["score"].astype(float)
tfidf_ridge.fit(X_train, y_train)

test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test_df = pd.read_csv(test_path)
test_df["full_text"] = test_df["full_text"].apply(preprocess_essay_text)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
LookupError                               Traceback (most recent call last)
/tmp/ipykernel_55/1738333705.py in <cell line: 0>()
     30 train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
     31 train_df = pd.read_csv(train_path)
---> 32 train_df["full_text"] = train_df["full_text"].apply(preprocess_essay_text)
     33 
     34 # Build a simple TF‑IDF + Ridge regression model

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_55/1738333705.py in preprocess_essay_text(text)
     20 
     21 def preprocess_essay_text(text: str) -> str:
---> 22     text = resolve_encodings_and_normalize(text)
     23     text = re.sub(r"\s+", " ", text.strip())
     24     text = re.sub(r'\s+([?.!,"])', r"\1", text)

/tmp/ipykernel_55/1738333705.py in resolve_encodings_and_normalize(text)
     11     text = (
     12         text.encode("raw_unicode_escape")
---> 13         .decode("utf-8", errors="replace_decoding_with_cp1252")
     14         .encode("cp1252", errors="replace_encoding_with_utf8")
     15         .decode("utf-8", errors="replace_decoding_with_cp1252")

LookupError: unknown error handler name 'replace_decoding_with_cp1252'

## === cell 1
test_predictions = tfidf_ridge.predict(test_df["full_text"].astype(str))
test_predictions = np.rint(test_predictions).astype(int)
test_predictions = np.clip(test_predictions, 1, 6)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3997727647.py in <cell line: 0>()
      1 # Predict scores for the test set
----> 2 test_predictions = tfidf_ridge.predict(test_df["full_text"].astype(str))
      3 # Round to nearest integer within the valid score range 1‑6
      4 test_predictions = np.rint(test_predictions).astype(int)
      5 test_predictions = np.clip(test_predictions, 1, 6)

NameError: name 'tfidf_ridge' is not defined

## === cell 2
submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission saved to 'submission.csv'")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1660407147.py in <cell line: 0>()
      1 # Prepare submission file
----> 2 submission = pd.DataFrame({"essay_id": test_df["essay_id"], "score": test_predictions})
      3 submission.to_csv("submission.csv", index=False)
      4 print("Submission saved to 'submission.csv'")

NameError: name 'test_df' is not defined
