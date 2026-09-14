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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

0.6623544245383161

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import glob
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np

print("Python:", sys.version)
print("Working dir:", os.getcwd())
print("Listing /kaggle/input (top-level):", os.listdir("/kaggle/input")[:10])



## === cell 1
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test_df = pd.read_csv(TEST_PATH)

test_df["full_text"] = (
    test_df["full_text"].astype(str).str.replace("\n", " ", regex=False)
)

test_df.head()



## === cell 2
out_tsv = "/kaggle/working/p8_fold3_test.txt"
test_df_out = test_df.copy()

if "score" not in test_df_out.columns:
    test_df_out["score"] = np.random.randint(
        1, 7, size=len(test_df_out)
    )  # inclusive 1..6

test_df_out.to_csv(out_tsv, header=False, index=False, sep="\t")
print("Wrote:", out_tsv, "rows:", len(test_df_out))



## === cell 3
MODEL_DIR = "/kaggle/input/multi-scale-bert-aes"
script_path = os.path.join(MODEL_DIR, "predict_multi_scale_multi_loss.py")
assert os.path.exists(script_path), f"Missing script: {script_path}"

before = set(glob.glob("/kaggle/working/*"))

print("Running external predictor...")
res = subprocess.run(
    [sys.executable, script_path], cwd=MODEL_DIR, capture_output=True, text=True
)
print("Return code:", res.returncode)
print("STDOUT (last 2000 chars):\n", res.stdout[-2000:])
print("STDERR (last 2000 chars):\n", res.stderr[-2000:])

if res.returncode != 0:
    raise RuntimeError("External predictor failed. See logs above.")

after = set(glob.glob("/kaggle/working/*"))
new_files = sorted(list(after - before))
print("New files created in /kaggle/working:", new_files)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/4089327378.py in <cell line: 0>()
      3 MODEL_DIR = "/kaggle/input/multi-scale-bert-aes"
      4 script_path = os.path.join(MODEL_DIR, "predict_multi_scale_multi_loss.py")
----> 5 assert os.path.exists(script_path), f"Missing script: {script_path}"
      6 
      7 # Snapshot /kaggle/working before running to find newly created prediction files.

AssertionError: Missing script: /kaggle/input/multi-scale-bert-aes/predict_multi_scale_multi_loss.py

## === cell 4
candidates = []

expected = "/kaggle/working/pred.csv"
if os.path.exists(expected):
    candidates.append(expected)

for pat in ["/kaggle/working/*.csv", "/kaggle/working/*.tsv", "/kaggle/working/*.txt"]:
    candidates.extend(sorted(glob.glob(pat)))

seen = set()
candidates = [c for c in candidates if not (c in seen or seen.add(c))]

print("Candidate prediction files:")
for c in candidates[:50]:
    print(" -", c)


def try_load_pred(path: str):
    for sep in ["\t", ","]:
        try:
            dfp = pd.read_csv(path, sep=sep, header=None)
            if dfp.shape[1] >= 2 and len(dfp) == len(test_df):
                dfp = dfp.iloc[:, :2].copy()
                dfp.columns = ["label", "pred"]
                return dfp, sep
        except Exception:
            pass
    return None, None


df_pred = None
used_path = None
used_sep = None
for c in candidates:
    dfp, sep = try_load_pred(c)
    if dfp is not None:
        df_pred = dfp
        used_path = c
        used_sep = sep
        break

if df_pred is None:
    raise FileNotFoundError(
        "Could not find a usable prediction file with the same number of rows as test set "
        f"({len(test_df)}). Check external script outputs in /kaggle/working."
    )

print(f"Loaded predictions from: {used_path} (sep='{used_sep}') shape={df_pred.shape}")
df_pred.head()



## === cell 5
thresh1 = 27.783113014764947
thresh2 = 32.743012411282415
thresh3 = 36.263925604171554
thresh4 = 39.169183811705835
thresh5 = 41.44527439514066


def round_pred_to_class(pred):
    if pred < thresh1:
        return 1
    elif pred < thresh2:
        return 2
    elif pred < thresh3:
        return 3
    elif pred < thresh4:
        return 4
    elif pred < thresh5:
        return 5
    else:
        return 6


df_pred["pred"] = pd.to_numeric(df_pred["pred"], errors="coerce")
if df_pred["pred"].isna().any():
    raise ValueError("Predictions contain non-numeric values after conversion.")

df_pred["pred_1to6"] = df_pred["pred"].apply(round_pred_to_class).astype(int)
df_pred[["pred", "pred_1to6"]].head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2804631095.py in <cell line: 0>()
     25 df_pred["pred"] = pd.to_numeric(df_pred["pred"], errors="coerce")
     26 if df_pred["pred"].isna().any():
---> 27     raise ValueError("Predictions contain non-numeric values after conversion.")
     28 
     29 df_pred["pred_1to6"] = df_pred["pred"].apply(round_pred_to_class).astype(int)

ValueError: Predictions contain non-numeric values after conversion.

## === cell 6
if len(df_pred) != len(test_df):
    raise ValueError(f"Row mismatch: test={len(test_df)} preds={len(df_pred)}")

submission = pd.DataFrame(
    {"essay_id": test_df["essay_id"].values, "score": df_pred["pred_1to6"].values}
)
submission["score"] = submission["score"].clip(1, 6).astype(int)

sub_path = "/kaggle/working/submission.csv"
submission.to_csv(sub_path, index=False)

print("Wrote submission:", sub_path)
print(submission.head())
print("Submission shape:", submission.shape)
print("Score value counts:\n", submission["score"].value_counts().sort_index())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'pred_1to6'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1302851302.py in <cell line: 0>()
      4 
      5 submission = pd.DataFrame(
----> 6     {"essay_id": test_df["essay_id"].values, "score": df_pred["pred_1to6"].values}
      7 )
      8 submission["score"] = submission["score"].clip(1, 6).astype(int)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'pred_1to6'
