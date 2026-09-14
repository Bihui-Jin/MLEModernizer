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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
h2o==3.46.0.8
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.7859834791377697

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49963) has done: 'I fix the failing data-loading step by switching from the missing `melanoma-train-test-creator` paths to the dataset paths that actually exist in this environment (`../input/siim-isic-melanoma-classification/train.csv` and `test.csv`). To preserve the core H2O AutoML approach, I keep the same training/prediction flow, but correct the feature list so it excludes the target (and any non-feature identifiers) to prevent leakage/invalid columns. I also ensure categorical columns are treated as factors and that missing values are handled consistently by H2O. Finally, I generate `submission.csv` with the required columns and align prediction rows to `image_name` to guarantee a valid Kaggle submission file.'
- What this solution (achieved 0.69414) has done: 'I fix the H2O data-loading failure by removing the unsupported `columns=` argument and instead loading the full CSV then subsetting to the required columns, which unblocks all downstream cells. I also ensure `patient_id` is treated as a categorical (string then factor) and align factor levels between train/test to prevent runtime issues at predict time. Finally, I keep the AutoML training flow the same but make sure we always extract the correct probability column (`p1` if present, otherwise the positive class column) and write a valid `submission.csv` with the required schema.'
- What this solution (achieved 0.5) has done: 'I fix the H2O AutoML crash caused by early-stopping tolerance auto-derivation dividing by zero, by removing custom early-stopping settings and letting AutoML use safe defaults (this is a minimal change and unblocks training). I also make the fold split robust by filtering using the correct factor level type (so the validation frame is non-empty), which prevents downstream issues and helps AUC. Finally, I keep the same feature set and AutoML flow, but add a safe fallback to generate a valid `submission.csv` even if AutoML unexpectedly fails, ensuring an end-to-end run that always writes the required file.'
- What this solution (achieved 0.5) has done: 'I fix the empty train/valid split by using the already-created `fold` factor column for filtering instead of the numeric `fold_num`, which currently sends all rows to validation. Then I address the H2O AutoML `/ by zero` server error by setting an explicit, safe `stopping_tolerance` so AutoML doesn’t try to auto-derive it from the frame. Finally, I keep the same H2O AutoML training/prediction flow but make the probability-column extraction more robust, ensuring we always write a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.5) has done: 'I fix the empty train/valid split by filtering on the numeric `fold_num` before it’s converted to a factor (your current comparison against string `"0"` makes every row go to validation). Then I address the H2O AutoML `/ by zero` crash by removing the explicit early-stopping tolerance override so AutoML uses its internal safe defaults (this keeps the same AutoML training flow, just avoids the server-side bug trigger). Finally, I keep the same prediction-to-submission pipeline but ensure we always extract the positive-class probability robustly and write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the empty train/valid split by creating `fold_num` as a proper numeric column and filtering on that numeric column (your current boolean mask is inverted due to H2O expression/type handling, sending all rows to validation). Then I address the H2O AutoML `/ by zero` crash by setting a safe explicit `stopping_tolerance` (keeping the same AutoML training flow, just preventing the server-side auto-derivation bug). Finally, I keep your prediction/submission logic intact but make it robust to label/probability column naming, ensuring we always write a valid `submission.csv` with the correct rows and columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os
import random

SEED = 47
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
import h2o

print(h2o.__version__)
from h2o.automl import H2OAutoML

h2o.init(max_mem_size="16G", nthreads=-1, log_level="WARN")



## === cell 2
TRAIN_CSV = "../input/siim-isic-melanoma-classification/train.csv"
TEST_CSV = "../input/siim-isic-melanoma-classification/test.csv"

y = "target"
id_col = "image_name"
drop_cols = {"diagnosis", "benign_malignant"}  # never used as features
feature_candidates = [
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
]

fold_col = "fold"
fold_num_col = "fold_num"

train_cols = [id_col, y] + feature_candidates
test_cols = [id_col] + feature_candidates

train_pd = pd.read_csv(TRAIN_CSV, usecols=train_cols)
test_pd_full = pd.read_csv(TEST_CSV, usecols=test_cols)

missing_train = [c for c in train_cols if c not in train_pd.columns]
missing_test = [c for c in test_cols if c not in test_pd_full.columns]
if missing_train:
    raise ValueError(f"Train CSV missing required columns: {missing_train}")
if missing_test:
    raise ValueError(f"Test CSV missing required columns: {missing_test}")



## === cell 3
non_features = {id_col, y} | drop_cols
x = [c for c in train_pd.columns if c not in non_features]

missing_in_test = [c for c in x if c not in test_pd_full.columns]
if missing_in_test:
    raise ValueError(f"Test set missing expected feature columns: {missing_in_test}")



## === cell 4
pid_series = (
    train_pd["patient_id"].astype(str)
    if "patient_id" in train_pd.columns
    else train_pd[id_col].astype(str)
)
pid_digits = pid_series.str.replace(r"[^0-9]", "", regex=True)
pid_num = pd.to_numeric(pid_digits, errors="coerce")
pid_len = pid_series.str.len().astype(float)
pid_num_filled = pid_num.fillna(pid_len)
fold_num = (pid_num_filled.astype(np.int64) % 5).astype(np.int8)

train_pd[fold_num_col] = fold_num
train_pd[fold_col] = train_pd[fold_num_col].astype("category")

train_pd_train = train_pd.loc[train_pd[fold_num_col] != 0].copy()
train_pd_valid = train_pd.loc[train_pd[fold_num_col] == 0].copy()

x = [c for c in x if c not in {fold_col, fold_num_col}]

print("Train rows:", len(train_pd_train), "Valid rows:", len(train_pd_valid))
if len(train_pd_train) == 0 or len(train_pd_valid) == 0:
    raise RuntimeError(
        f"Invalid split produced empty frame(s): train={len(train_pd_train)}, valid={len(train_pd_valid)}"
    )

train = h2o.H2OFrame(train_pd)
test = h2o.H2OFrame(test_pd_full)

train_frame = h2o.H2OFrame(train_pd_train)
valid_frame = h2o.H2OFrame(train_pd_valid)

train_frame[y] = train_frame[y].asfactor()
valid_frame[y] = valid_frame[y].asfactor()

for c in ["sex", "anatom_site_general_challenge", "patient_id"]:
    if c in train_frame.columns and c in test.columns:
        train_frame[c] = train_frame[c].asfactor()
        valid_frame[c] = valid_frame[c].asfactor()
        test[c] = test[c].asfactor()
        lvls = train_frame[c].levels()
        if isinstance(lvls, list) and len(lvls) == 1 and isinstance(lvls[0], list):
            lvls = lvls[0]
        test[c] = test[c].set_levels(lvls)

if "age_approx" in train_frame.columns:
    train_frame["age_approx"] = train_frame["age_approx"].asnumeric()
    valid_frame["age_approx"] = valid_frame["age_approx"].asnumeric()
    test["age_approx"] = test["age_approx"].asnumeric()



## === cell 5
aml = H2OAutoML(
    max_models=30,
    seed=SEED,
    max_runtime_secs=600,
    sort_metric="AUC",
    nfolds=0,
    keep_cross_validation_predictions=False,
    keep_cross_validation_models=False,
    keep_cross_validation_fold_assignment=False,
)
aml.train(x=x, y=y, training_frame=train_frame, validation_frame=valid_frame)



## === cell 6
lb = aml.leaderboard
print(lb.head(rows=min(10, lb.nrows)))



## === cell 7
leader = aml.leader
if leader is None:
    raise RuntimeError(
        "AutoML did not produce a leader model; cannot proceed to prediction."
    )



## === cell 8
preds = leader.predict(test)

pred_cols = preds.columns
if "p1" in pred_cols:
    prob_col = "p1"
elif "1" in pred_cols:
    prob_col = "1"
elif "pTrue" in pred_cols:
    prob_col = "pTrue"
elif "True" in pred_cols:
    prob_col = "True"
else:
    if "predict" in pred_cols and len(pred_cols) >= 3:
        prob_col = pred_cols[-1]
    else:
        raise ValueError(f"Unexpected prediction frame columns: {pred_cols}")

pred_p1 = h2o.as_list(preds[prob_col], use_pandas=True)
pred_p1.columns = ["target"]



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
H2OResponseError                          Traceback (most recent call last)
/tmp/ipykernel_11/3515358060.py in <cell line: 0>()
----> 1 preds = leader.predict(test)
      2 
      3 pred_cols = preds.columns
      4 if "p1" in pred_cols:
      5     prob_col = "p1"

/usr/local/lib/python3.11/dist-packages/h2o/model/model_base.py in predict(self, test_data, custom_metric, custom_metric_func)
    330             eval_func_ref = h2o.upload_custom_metric(custom_metric)
    331         if not isinstance(test_data, h2o.H2OFrame): raise ValueError("test_data must be an instance of H2OFrame")
--> 332         j = H2OJob(h2o.api("POST /4/Predictions/models/%s/frames/%s" % (self.model_id, test_data.frame_id), data = {'custom_metric_func': custom_metric_func}),
    333                    self._model_json["algo"] + " prediction")
    334         j.poll()

/usr/local/lib/python3.11/dist-packages/h2o/frame.py in frame_id(self)
    412         >>> print(iris.frame_id)
    413         """
--> 414         return self._frame()._ex._cache._id
    415 
    416     @frame_id.setter

/usr/local/lib/python3.11/dist-packages/h2o/frame.py in _frame(self, rows, rows_offset, cols, cols_offset, fill_cache)
    582 
    583     def _frame(self, rows=10, rows_offset=0, cols=-1, cols_offset=0, fill_cache=False):
--> 584         self._ex._eager_frame()
    585         if fill_cache:
    586             self._ex._cache.fill(rows=rows, rows_offset=rows_offset, cols=cols, cols_offset=cols_offset)

/usr/local/lib/python3.11/dist-packages/h2o/expr.py in _eager_frame(self)
     88         if not self._cache.is_empty(): return
     89         if self._cache._id is not None: return  # Data already computed under ID, but not cached locally
---> 90         self._eval_driver('frame')
     91 
     92     def _eager_scalar(self):  # returns a scalar (or a list of scalars)

/usr/local/lib/python3.11/dist-packages/h2o/expr.py in _eval_driver(self, top)
    112         """
    113         exec_str = self._get_ast_str(top)
--> 114         res = ExprNode.rapids(exec_str)
    115         if 'scalar' in res:
    116             if isinstance(res['scalar'], list):

/usr/local/lib/python3.11/dist-packages/h2o/expr.py in rapids(expr)
    256         :returns: The JSON response (as a python dictionary) of the Rapids execution
    257         """
--> 258         return h2o.api("POST /99/Rapids", data={"ast": expr, "session_id": h2o.connection().session_id})
    259 
    260 

/usr/local/lib/python3.11/dist-packages/h2o/h2o.py in api(endpoint, data, json, filename, save_to)
    121     # type checks are performed in H2OConnection class
    122     _check_connection()
--> 123     return h2oconn.request(endpoint, data=data, json=json, filename=filename, save_to=save_to)
    124 
    125 

/usr/local/lib/python3.11/dist-packages/h2o/backend/connection.py in request(self, endpoint, data, json, filename, save_to)
    497                     save_to = save_to(resp)
    498                 self._log_end_transaction(start_time, resp)
--> 499                 return self._process_response(resp, save_to)
    500 
    501             except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:

/usr/local/lib/python3.11/dist-packages/h2o/backend/connection.py in _process_response(response, save_to)
    851         if status_code in {400, 404, 412} and isinstance(data, H2OErrorV3):
    852             data.show_stacktrace = False
--> 853             raise H2OResponseError(data)
    854 
    855         # Server errors (notably 500 = "Server Error")

H2OResponseError: Server error java.lang.IllegalArgumentException:
  Error: Number of replacement factors must equal current number of levels. Current number of levels: 1457 != 1665
  Request: POST /99/Rapids
    data: {'ast': "(tmp= py_27_sid_96af (:= (tmp= py_26_sid_96af (:= (tmp= py_25_sid_96af (:= (tmp= py_24_sid_96af (:= (tmp= py_23_sid_96af (:= (tmp= py_22_sid_96af (:= (tmp= py_21_sid_96af (:= Key_Frame__upload_9e40a37ee4a39cb26f16a6ff6a83464f.hex (as.factor (cols_py Key_Frame__upload_9e40a37ee4a39cb26f16a6ff6a83464f.hex 'sex')) 2 [])) (setDomain (cols_py py_21_sid_96af 'sex') False ['female' 'male' 'nan']) 2 [])) (as.factor (cols_py py_22_sid_96af 'anatom_site_general_challenge')) 4 [])) (setDomain (cols_py py_23_sid_96af 'anatom_site_general_challenge') False ['head/neck' 'lower extremity' 'nan' 'oral/genital' 'palms/soles' 'torso' 'upper extremity']) 4 [])) (as.factor (cols_py py_24_sid_96af 'patient_id')) 1 [])) (setDomain (cols_py py_25_sid_96af 'patient_id') False ['IP_0019713' 'IP_0036322' 'IP_0038436' 'IP_0039318' 'IP_0045462' 'IP_0050531' 'IP_0059113' 'IP_0063457' 'IP_0063782' 'IP_0065733' 'IP_0067018' 'IP_0070552' 'IP_0074589' 'IP_0078294' 'IP_0078763' 'IP_0079464' 'IP_0085748' 'IP_0093378' 'IP_0096763' 'IP_0097257' 'IP_0099096' 'IP_0099726' 'IP_0108601' 'IP_0122358' 'IP_0128628' 'IP_0135517' 'IP_0139601' 'IP_0147446' 'IP_0148558' 'IP_0148851' 'IP_0156057' 'IP_0161742' 'IP_0170821' 'IP_0175414' 'IP_0175539' 'IP_0197769' 'IP_0232771' 'IP_0243839' 'IP_0248387' 'IP_0280959' 'IP_0301948' 'IP_0306499' 'IP_0307863' 'IP_0347699' 'IP_0353119' 'IP_0358262' 'IP_0358698' 'IP_0367464' 'IP_0387543' 'IP_0391953' 'IP_0392287' 'IP_0393526' 'IP_0394288' 'IP_0398596' 'IP_0400168' 'IP_0400826' 'IP_0408084' 'IP_0408346' 'IP_0410088' 'IP_0412224' 'IP_0414408' 'IP_0418246' 'IP_0423032' 'IP_0428413' 'IP_0442139' 'IP_0444327' 'IP_0446831' 'IP_0454573' 'IP_0466878' 'IP_0475504' 'IP_0483228' 'IP_0492456' 'IP_0500536' 'IP_0500903' 'IP_0503166' 'IP_0507923' 'IP_0512752' 'IP_0516326' 'IP_0517623' 'IP_0521041' 'IP_0524772' 'IP_0536839' 'IP_0540298' 'IP_0541183' 'IP_0550106' 'IP_0550778' 'IP_0583343' 'IP_0585473' 'IP_0593168' 'IP_0612651' 'IP_0615694' 'IP_0621614' 'IP_0623861' 'IP_0630349' 'IP_0634841' 'IP_0637703' 'IP_0639877' 'IP_0648521' 'IP_0656529' 'IP_0657134' 'IP_0663261' 'IP_0672967' 'IP_0673451' 'IP_0677357' 'IP_0687884' 'IP_0689197' 'IP_0690987' 'IP_0699776' 'IP_0703787' 'IP_0716558' 'IP_0718832' 'IP_0725027' 'IP_0738123' 'IP_0760108' 'IP_0765629' 'IP_0768507' 'IP_0776416' 'IP_0777867' 'IP_0784448' 'IP_0787478' 'IP_0801043' 'IP_0825081' 'IP_0828404' 'IP_0835341' 'IP_0843056' 'IP_0843629' 'IP_0855064' 'IP_0857646' 'IP_0859301' 'IP_0867894' 'IP_0877882' 'IP_0878241' 'IP_0883591' 'IP_0891666' 'IP_0892152' 'IP_0896904' 'IP_0900824' 'IP_0904259' 'IP_0915233' 'IP_0931383' 'IP_0933819' 'IP_0934857' 'IP_0942159' 'IP_0951529' 'IP_0951571' 'IP_0957064' 'IP_0963996' 'IP_0974501' 'IP_0986794' 'IP_0989858' 'IP_0992551' 'IP_0998852' 'IP_1005683' 'IP_1005794' 'IP_1010643' 'IP_1023029' 'IP_1036531' 'IP_1039004' 'IP_1039996' 'IP_1043359' 'IP_1051664' 'IP_1058691' 'IP_1066187' 'IP_1066748' 'IP_1077703' 'IP_1091619' 'IP_1096358' 'IP_1106509' 'IP_1109756' 'IP_1132071' 'IP_1135236' 'IP_1136024' 'IP_1139701' 'IP_1140144' 'IP_1152723' 'IP_1157267' 'IP_1165806' 'IP_1170412' 'IP_1172238' 'IP_1173567' 'IP_1179846' 'IP_1184509' 'IP_1187676' 'IP_1188709' 'IP_1195596' 'IP_1197949' 'IP_1207677' 'IP_1208141' 'IP_1210062' 'IP_1211926' 'IP_1214216' 'IP_1218338' 'IP_1222917' 'IP_1230004' 'IP_1233669' 'IP_1242186' 'IP_1244021' 'IP_1246698' 'IP_1264754' 'IP_1273286' 'IP_1300691' 'IP_1303857' 'IP_1305429' 'IP_1310148' 'IP_1330994' 'IP_1334501' 'IP_1348848' 'IP_1351218' 'IP_1355716' 'IP_1355796' 'IP_1357049' 'IP_1362494' 'IP_1380269' 'IP_1386843' 'IP_1386934' 'IP_1395856' 'IP_1399166' 'IP_1418292' 'IP_1419269' 'IP_1420122' 'IP_1422713' 'IP_1447682' 'IP_1454551' 'IP_1470733' 'IP_1472634' 'IP_1476317' 'IP_1477087' 'IP_1481008' 'IP_1495267' 'IP_1504836' 'IP_1512731' 'IP_1512936' 'IP_1514144' 'IP_1514334' 'IP_1515878' 'IP_1517386' 'IP_1522453' 'IP_1525419' 'IP_1525623' 'IP_1533126' 'IP_1545708' 'IP_1557123' 'IP_1559911' 'IP_1559941' 'IP_1564049' 'IP_1564172' 'IP_1567348' 'IP_1568923' 'IP_1576187' 'IP_1582457' 'IP_1583003' 'IP_1583136' 'IP_1587853' 'IP_1599082' 'IP_1600028' 'IP_1613116' 'IP_1615189' 'IP_1627142' 'IP_1633296' 'IP_1636784' 'IP_1645936' 'IP_1652699' 'IP_1657504' 'IP_1657778' 'IP_1658752' 'IP_1671181' 'IP_1672532' 'IP_1672936' 'IP_1676499' 'IP_1681872' 'IP_1705144' 'IP_1723924' 'IP_1728609' 'IP_1730314' 'IP_1743439' 'IP_1751467' 'IP_1769689' 'IP_1787329' 'IP_1796383' 'IP_1800426' 'IP_1807952' 'IP_1822686' 'IP_1840599' 'IP_1849461' 'IP_1859257' 'IP_1866634' 'IP_1870306' 'IP_1885773' 'IP_1890779' 'IP_1902399' 'IP_1902763' 'IP_1915916' 'IP_1916588' 'IP_1920963' 'IP_1921028' 'IP_1976214' 'IP_1981261' 'IP_1982047' 'IP_1989404' 'IP_1991877' 'IP_1994743' 'IP_1999627' 'IP_2005898' 'IP_2006089' 'IP_2008749' 'IP_2010919' 'IP_2014901' 'IP_2019626' 'IP_2020141' 'IP_2026598' 'IP_2039882' 'IP_2040538' 'IP_2041319' 'IP_2043846' 'IP_2047697' 'IP_2049351' 'IP_2055758' 'IP_2075804' 'IP_2085351' 'IP_2102792' 'IP_2105363' 'IP_2107539' 'IP_2107993' 'IP_2109364' 'IP_2117218' 'IP_2139337' 'IP_2139548' 'IP_2143646' 'IP_2147407' 'IP_2152204' 'IP_2153088' 'IP_2170331' 'IP_2172761' 'IP_2177417' 'IP_2187484' 'IP_2189124' 'IP_2213648' 'IP_2214113' 'IP_2228564' 'IP_2250929' 'IP_2251646' 'IP_2263216' 'IP_2263888' 'IP_2278509' 'IP_2285751' 'IP_2302748' 'IP_2310783' 'IP_2317609' 'IP_2318163' 'IP_2321962' 'IP_2322991' 'IP_2325148' 'IP_2350511' 'IP_2358028' 'IP_2360609' 'IP_2375279' 'IP_2375994' 'IP_2378734' 'IP_2390008' 'IP_2394927' 'IP_2410124' 'IP_2411134' 'IP_2412574' 'IP_2422802' 'IP_2426108' 'IP_2428722' 'IP_2436697' 'IP_2442926' 'IP_2443623' 'IP_2444029' 'IP_2452694' 'IP_2455242' 'IP_2457893' 'IP_2465701' 'IP_2479386' 'IP_2482649' 'IP_2489908' 'IP_2501837' 'IP_2501903' 'IP_2506801' 'IP_2506892' 'IP_2507136' 'IP_2507276' 'IP_2508654' 'IP_2511209' 'IP_2516168' 'IP_2520657' 'IP_2538906' 'IP_2545631' 'IP_2548997' 'IP_2558686' 'IP_2575279' 'IP_2577172' 'IP_2584339' 'IP_2594106' 'IP_2606884' 'IP_2610032' 'IP_2613332' 'IP_2613684' 'IP_2617043' 'IP_2618037' 'IP_2619879' 'IP_2626937' 'IP_2632866' 'IP_2644876' 'IP_2654837' 'IP_2661172' 'IP_2665826' 'IP_2666042' 'IP_2669371' 'IP_2669703' 'IP_2675593' 'IP_2676056' 'IP_2679808' 'IP_2681677' 'IP_2685241' 'IP_2694513' 'IP_2697411' 'IP_2707228' 'IP_2724119' 'IP_2725676' 'IP_2732171' 'IP_2738587' 'IP_2746444' 'IP_2752722' 'IP_2760044' 'IP_2766521' 'IP_2769817' 'IP_2772363' 'IP_2785084' 'IP_2789041' 'IP_2793798' 'IP_2804368' 'IP_2809878' 'IP_2810418' 'IP_2817734' 'IP_2821147' 'IP_2821754' 'IP_2822147' 'IP_2823211' 'IP_2823659' 'IP_2825529' 'IP_2835119' 'IP_2838638' 'IP_2842074' 'IP_2842809' 'IP_2845131' 'IP_2850771' 'IP_2852433' 'IP_2853271' 'IP_2865063' 'IP_2869113' 'IP_2871134' 'IP_2873493' 'IP_2877278' 'IP_2881084' 'IP_2898839' 'IP_2900789' 'IP_2908137' 'IP_2913292' 'IP_2921424' 'IP_2922983' 'IP_2932439' 'IP_2934687' 'IP_2938573' 'IP_2947236' 'IP_2961528' 'IP_2962626' 'IP_2963366' 'IP_2978129' 'IP_2986217' 'IP_2987599' 'IP_2989764' 'IP_2999492' 'IP_3008149' 'IP_3010556' 'IP_3027709' 'IP_3034671' 'IP_3045056' 'IP_3047844' 'IP_3048629' 'IP_3054486' 'IP_3055163' 'IP_3055814' 'IP_3057277' 'IP_3067471' 'IP_3075186' 'IP_3078108' 'IP_3081924' 'IP_3089004' 'IP_3091321' 'IP_3094561' 'IP_3110826' 'IP_3112726' 'IP_3116217' 'IP_3120589' 'IP_3127942' 'IP_3128221' 'IP_3140047' 'IP_3141271' 'IP_3145948' 'IP_3155633' 'IP_3169043' 'IP_3172139' 'IP_3190947' 'IP_3191079' 'IP_3191158' 'IP_3219832' 'IP_3222187' 'IP_3228049' 'IP_3230778' 'IP_3232631' 'IP_3237442' 'IP_3237448' 'IP_3240837' 'IP_3246668' 'IP_3248713' 'IP_3255507' 'IP_3260326' 'IP_3264929' 'IP_3281766' 'IP_3293337' 'IP_3293726' 'IP_3298186' 'IP_3315536' 'IP_3324603' 'IP_3335629' 'IP_3336327' 'IP_3349523' 'IP_3350187' 'IP_3358218' 'IP_3359673' 'IP_3365162' 'IP_3370847' 'IP_3395413' 'IP_3397861' 'IP_3402498' 'IP_3419066' 'IP_3419097' 'IP_3420228' 'IP_3422736' 'IP_3423256' 'IP_3424938' 'IP_3432258' 'IP_3433747' 'IP_3444084' 'IP_3448782' 'IP_3449666' 'IP_3454417' 'IP_3466191' 'IP_3468257' 'IP_3473819' 'IP_3474679' 'IP_3490992' 'IP_3511266' 'IP_3511494' 'IP_3512888' 'IP_3517456' 'IP_3522576' 'IP_3523452' 'IP_3523549' 'IP_3531652' 'IP_3531667' 'IP_3534839' 'IP_3537591' 'IP_3541322' 'IP_3541879' 'IP_3549978' 'IP_3557056' 'IP_3561033' 'IP_3561168' 'IP_3561719' 'IP_3562983' 'IP_3573146' 'IP_3575311' 'IP_3576583' 'IP_3586223' 'IP_3587277' 'IP_3603767' 'IP_3609743' 'IP_3624052' 'IP_3631578' 'IP_3635573' 'IP_3636528' 'IP_3637671' 'IP_3645134' 'IP_3658607' 'IP_3659172' 'IP_3662918' 'IP_3673974' 'IP_3675183' 'IP_3676113' 'IP_3678398' 'IP_3679859' 'IP_3686214' 'IP_3687816' 'IP_3689742' 'IP_3690477' 'IP_3691058' 'IP_3693916' 'IP_3699526' 'IP_3701282' 'IP_3707904' 'IP_3710233' 'IP_3713422' 'IP_3714517' 'IP_3722746' 'IP_3728091' 'IP_3729287' 'IP_3729342' 'IP_3735763' 'IP_3739659' 'IP_3749901' 'IP_3750016' 'IP_3751991' 'IP_3752129' 'IP_3755407' 'IP_3776424' 'IP_3776544' 'IP_3780641' 'IP_3785646' 'IP_3789216' 'IP_3789586' 'IP_3789903' 'IP_3807016' 'IP_3814907' 'IP_3846489' 'IP_3854976' 'IP_3857996' 'IP_3858753' 'IP_3860796' 'IP_3863213' 'IP_3865267' 'IP_3868067' 'IP_3882396' 'IP_3882858' 'IP_3893748' 'IP_3897989' 'IP_3908059' 'IP_3917378' 'IP_3927229' 'IP_3927686' 'IP_3927998' 'IP_3933152' 'IP_3937212' 'IP_3938557' 'IP_3941296' 'IP_3943416' 'IP_3947094' 'IP_3954109' 'IP_3970734' 'IP_3984729' 'IP_3986523' 'IP_3994607' 'IP_4005662' 'IP_4009777' 'IP_4015774' 'IP_4017726' 'IP_4021847' 'IP_4030464' 'IP_4032184' 'IP_4042098' 'IP_4066898' 'IP_4070933' 'IP_4078063' 'IP_4080851' 'IP_4094708' 'IP_4095099' 'IP_4096093' 'IP_4098846' 'IP_4103436' 'IP_4103643' 'IP_4109313' 'IP_4110621' 'IP_4112074' 'IP_4116267' 'IP_4117284' 'IP_4122224' 'IP_4122606' 'IP_4129058' 'IP_4137989' 'IP_4150634' 'IP_4154441' 'IP_4162876' 'IP_4169963' 'IP_4173362' 'IP_4179834' 'IP_4194721' 'IP_4199752' 'IP_4204657' 'IP_4208266' 'IP_4210911' 'IP_4211276' 'IP_4218429' 'IP_4234031' 'IP_4235783' 'IP_4248414' 'IP_4251788' 'IP_4252478' 'IP_4254406' 'IP_4263066' 'IP_4267931' 'IP_4277352' 'IP_4281194' 'IP_4292424' 'IP_4294672' 'IP_4295309' 'IP_4297063' 'IP_4297939' 'IP_4302652' 'IP_4311593' 'IP_4314793' 'IP_4315523' 'IP_4321822' 'IP_4342801' 'IP_4349196' 'IP_4353359' 'IP_4356379' 'IP_4362664' 'IP_4366219' 'IP_4366959' 'IP_4370559' 'IP_4379852' 'IP_4387709' 'IP_4389876' 'IP_4391034' 'IP_4409916' 'IP_4416742' 'IP_4424647' 'IP_4427711' 'IP_4436071' 'IP_4445194' 'IP_4448203' 'IP_4450258' 'IP_4453923' 'IP_4454844' 'IP_4462726' 'IP_4466046' 'IP_4477281' 'IP_4479736' 'IP_4488328' 'IP_4494551' 'IP_4520332' 'IP_4525077' 'IP_4538764' 'IP_4544162' 'IP_4548568' 'IP_4548593' 'IP_4557152' 'IP_4557431' 'IP_4558019' 'IP_4572846' 'IP_4576977' 'IP_4584109' 'IP_4604869' 'IP_4609144' 'IP_4611463' 'IP_4612042' 'IP_4615697' 'IP_4617831' 'IP_4624343' 'IP_4633933' 'IP_4638162' 'IP_4639224' 'IP_4645992' 'IP_4667127' 'IP_4669427' 'IP_4669931' 'IP_4681183' 'IP_4682288' 'IP_4698288' 'IP_4707488' 'IP_4711011' 'IP_4713174' 'IP_4716377' 'IP_4719884' 'IP_4721139' 'IP_4721772' 'IP_4722284' 'IP_4734386' 'IP_4745288' 'IP_4749521' 'IP_4750282' 'IP_4762742' 'IP_4765519' 'IP_4783134' 'IP_4795384' 'IP_4811352' 'IP_4821632' 'IP_4837579' 'IP_4841609' 'IP_4852531' 'IP_4871171' 'IP_4872151' 'IP_4880648' 'IP_4897162' 'IP_4898383' 'IP_4921034' 'IP_4936463' 'IP_4938349' 'IP_4938382' 'IP_4938438' 'IP_4956597' 'IP_4956722' 'IP_4959878' 'IP_4966841' 'IP_4969172' 'IP_4978372' 'IP_4982808' 'IP_4983809' 'IP_4988871' 'IP_4996313' 'IP_5004338' 'IP_5017621' 'IP_5027399' 'IP_5031178' 'IP_5031568' 'IP_5033731' 'IP_5037503' 'IP_5048227' 'IP_5063756' 'IP_5064161' 'IP_5064631' 'IP_5068447' 'IP_5071744' 'IP_5075533' 'IP_5076578' 'IP_5085022' 'IP_5086592' 'IP_5086698' 'IP_5096058' 'IP_5097631' 'IP_5109559' 'IP_5110222' 'IP_5111058' 'IP_5142207' 'IP_5152468' 'IP_5167473' 'IP_5168777' 'IP_5169409' 'IP_5174964' 'IP_5176431' 'IP_5176722' 'IP_5177184' 'IP_5178963' 'IP_5181027' 'IP_5188822' 'IP_5203658' 'IP_5205991' 'IP_5208504' 'IP_5229014' 'IP_5243587' 'IP_5250256' 'IP_5278162' 'IP_5287467' 'IP_5294508' 'IP_5295861' 'IP_5298487' 'IP_5304634' 'IP_5311557' 'IP_5318792' 'IP_5328472' 'IP_5331792' 'IP_5336201' 'IP_5339938' 'IP_5350484' 'IP_5357759' 'IP_5365464' 'IP_5368914' 'IP_5370714' 'IP_5384571' 'IP_5389392' 'IP_5390682' 'IP_5392943' 'IP_5397868' 'IP_5399626' 'IP_5408122' 'IP_5410936' 'IP_5424227' 'IP_5431521' 'IP_5438943' 'IP_5439716' 'IP_5440659' 'IP_5450817' 'IP_5461043' 'IP_5474292' 'IP_5489178' 'IP_5496141' 'IP_5496892' 'IP_5517026' 'IP_5519306' 'IP_5533537' 'IP_5537648' 'IP_5548623' 'IP_5550558' 'IP_5552354' 'IP_5568762' 'IP_5596716' 'IP_5598284' 'IP_5604493' 'IP_5605474' 'IP_5609476' 'IP_5621584' 'IP_5623232' 'IP_5627853' 'IP_5629228' 'IP_5638009' 'IP_5639149' 'IP_5649357' 'IP_5686827' 'IP_5691366' 'IP_5702642' 'IP_5702829' 'IP_5703282' 'IP_5711917' 'IP_5715797' 'IP_5724618' 'IP_5724756' 'IP_5724759' 'IP_5744544' 'IP_5752342' 'IP_5765893' 'IP_5766574' 'IP_5767581' 'IP_5768092' 'IP_5772279' 'IP_5773889' 'IP_5776149' 'IP_5793596' 'IP_5805281' 'IP_5808038' 'IP_5820702' 'IP_5830187' 'IP_5851912' 'IP_5852997' 'IP_5857448' 'IP_5865682' 'IP_5867829' 'IP_5874294' 'IP_5881171' 'IP_5881758' 'IP_5889408' 'IP_5890334' 'IP_5901901' 'IP_5903849' 'IP_5903912' 'IP_5906964' 'IP_5916797' 'IP_5917089' 'IP_5922049' 'IP_5940197' 'IP_5945051' 'IP_5945442' 'IP_5945814' 'IP_5972253' 'IP_5972548' 'IP_5974691' 'IP_5986216' 'IP_5987487' 'IP_6001986' 'IP_6002082' 'IP_6006409' 'IP_6008814' 'IP_6009973' 'IP_6017019' 'IP_6017204' 'IP_6021072' 'IP_6026359' 'IP_6029068' 'IP_6029234' 'IP_6029629' 'IP_6031322' 'IP_6051587' 'IP_6058804' 'IP_6061014' 'IP_6067392' 'IP_6071452' 'IP_6074542' 'IP_6078411' 'IP_6096257' 'IP_6098503' 'IP_6116226' 'IP_6118207' 'IP_6120178' 'IP_6121241' 'IP_6128767' 'IP_6129542' 'IP_6130936' 'IP_6134378' 'IP_6141131' 'IP_6142746' 'IP_6152307' 'IP_6153406' 'IP_6175417' 'IP_6204577' 'IP_6207062' 'IP_6212064' 'IP_6215369' 'IP_6219396' 'IP_6227894' 'IP_6228063' 'IP_6233127' 'IP_6234053' 'IP_6235676' 'IP_6242332' 'IP_6245507' 'IP_6247924' 'IP_6251534' 'IP_6261583' 'IP_6267856' 'IP_6271703' 'IP_6272363' 'IP_6275614' 'IP_6285147' 'IP_6293754' 'IP_6294394' 'IP_6297463' 'IP_6312474' 'IP_6318212' 'IP_6323321' 'IP_6326286' 'IP_6331798' 'IP_6332526' 'IP_6332843' 'IP_6341009' 'IP_6342052' 'IP_6342369' 'IP_6346671' 'IP_6349217' 'IP_6352254' 'IP_6374803' 'IP_6387073' 'IP_6397153' 'IP_6405528' 'IP_6415012' 'IP_6419409' 'IP_6420568' 'IP_6428232' 'IP_6433037' 'IP_6443758' 'IP_6445643' 'IP_6450431' 'IP_6470089' 'IP_6475387' 'IP_6486706' 'IP_6489379' 'IP_6496337' 'IP_6506834' 'IP_6510081' 'IP_6514499' 'IP_6526534' 'IP_6531672' 'IP_6532463' 'IP_6533286' 'IP_6538458' 'IP_6539844' 'IP_6541466' 'IP_6557127' 'IP_6567258' 'IP_6569383' 'IP_6572129' 'IP_6594273' 'IP_6599517' 'IP_6601134' 'IP_6601306' 'IP_6610197' 'IP_6613083' 'IP_6614604' 'IP_6616887' 'IP_6623396' 'IP_6625293' 'IP_6630831' 'IP_6648913' 'IP_6658881' 'IP_6661651' 'IP_6676823' 'IP_6678597' 'IP_6692103' 'IP_6694806' 'IP_6709202' 'IP_6715144' 'IP_6724798' 'IP_6730678' 'IP_6739626' 'IP_6764319' 'IP_6767827' 'IP_6776978' 'IP_6777579' 'IP_6795979' 'IP_6796539' 'IP_6808872' 'IP_6814066' 'IP_6814737' 'IP_6820518' 'IP_6830809' 'IP_6833889' 'IP_6835202' 'IP_6838833' 'IP_6842204' 'IP_6842734' 'IP_6850143' 'IP_6866228' 'IP_6868151' 'IP_6873524' 'IP_6875077' 'IP_6885799' 'IP_6887203' 'IP_6887429' 'IP_6888021' 'IP_6889268' 'IP_6889672' 'IP_6891522' 'IP_6892288' 'IP_6894159' 'IP_6901441' 'IP_6908816' 'IP_6911447' 'IP_6931421' 'IP_6932031' 'IP_6936664' 'IP_6939959' 'IP_6944941' 'IP_6945048' 'IP_6949859' 'IP_6960402' 'IP_6977878' 'IP_6981218' 'IP_6982152' 'IP_6982571' 'IP_6992078' 'IP_6997528' 'IP_7004499' 'IP_7005083' 'IP_7021484' 'IP_7037594' 'IP_7040211' 'IP_7050886' 'IP_7066637' 'IP_7069391' 'IP_7087967' 'IP_7096079' 'IP_7100012' 'IP_7104534' 'IP_7109467' 'IP_7111848' 'IP_7113209' 'IP_7121757' 'IP_7123023' 'IP_7123416' 'IP_7125941' 'IP_7130762' 'IP_7132179' 'IP_7132551' 'IP_7141746' 'IP_7142641' 'IP_7145044' 'IP_7147862' 'IP_7150378' 'IP_7150522' 'IP_7151331' 'IP_7160012' 'IP_7167373' 'IP_7198796' 'IP_7199022' 'IP_7203968' 'IP_7204032' 'IP_7211398' 'IP_7229146' 'IP_7240556' 'IP_7241377' 'IP_7243781' 'IP_7250026' 'IP_7252102' 'IP_7261254' 'IP_7262258' 'IP_7270568' 'IP_7279218' 'IP_7279968' 'IP_7291657' 'IP_7303276' 'IP_7318404' 'IP_7330399' 'IP_7334384' 'IP_7335423' 'IP_7341133' 'IP_7344936' 'IP_7345731' 'IP_7364944' 'IP_7371611' 'IP_7373371' 'IP_7375528' 'IP_7377609' 'IP_7382179' 'IP_7386073' 'IP_7388762' 'IP_7397472' 'IP_7403252' 'IP_7404837' 'IP_7404937' 'IP_7406922' 'IP_7412558' 'IP_7415009' 'IP_7415269' 'IP_7429637' 'IP_7442038' 'IP_7447953' 'IP_7452391' 'IP_7453424' 'IP_7456813' 'IP_7460121' 'IP_7466144' 'IP_7468494' 'IP_7471612' 'IP_7476769' 'IP_7481506' 'IP_7491908' 'IP_7497232' 'IP_7498456' 'IP_7499536' 'IP_7507194' 'IP_7507212' 'IP_7518373' 'IP_7522397' 'IP_7530983' 'IP_7548727' 'IP_7557262' 'IP_7562457' 'IP_7576319' 'IP_7584179' 'IP_7587497' 'IP_7594604' 'IP_7598598' 'IP_7599389' 'IP_7600629' 'IP_7610036' 'IP_7622888' 'IP_7629851' 'IP_7633542' 'IP_7644786' 'IP_7645011' 'IP_7647187' 'IP_7651698' 'IP_7665112' 'IP_7676978' 'IP_7685109' 'IP_7688669' 'IP_7688774' 'IP_7690396' 'IP_7691333' 'IP_7696681' 'IP_7700121' 'IP_7702038' 'IP_7716159' 'IP_7716626' 'IP_7718399' 'IP_7729044' 'IP_7735373' 'IP_7743081' 'IP_7743248' 'IP_7743588' 'IP_7748972' 'IP_7753447' 'IP_7765741' 'IP_7770083' 'IP_7772951' 'IP_7775567' 'IP_7782559' 'IP_7785592' 'IP_7792107' 'IP_7804786' 'IP_7817798' 'IP_7822951' 'IP_7824066' 'IP_7826053' 'IP_7826462' 'IP_7828866' 'IP_7829228' 'IP_7836372' 'IP_7838491' 'IP_7842431' 'IP_7854426' 'IP_7866394' 'IP_7868912' 'IP_7873698' 'IP_7887363' 'IP_7893332' 'IP_7894646' 'IP_7911457' 'IP_7926174' 'IP_7931229' 'IP_7932703' 'IP_7940004' 'IP_7946279' 'IP_7948039' 'IP_7950112' 'IP_7976842' 'IP_7979366' 'IP_7984337' 'IP_7991448' 'IP_7997841' 'IP_8001428' 'IP_8003017' 'IP_8003098' 'IP_8004532' 'IP_8007888' 'IP_8011614' 'IP_8019632' 'IP_8020102' 'IP_8027787' 'IP_8031662' 'IP_8039328' 'IP_8039381' 'IP_8041141' 'IP_8051701' 'IP_8069409' 'IP_8091497' 'IP_8094083' 'IP_8094961' 'IP_8101933' 'IP_8111876' 'IP_8124488' 'IP_8124898' 'IP_8130803' 'IP_8131348' 'IP_8132023' 'IP_8135032' 'IP_8136122' 'IP_8137203' 'IP_8140841' 'IP_8142299' 'IP_8142871' 'IP_8152038' 'IP_8154616' 'IP_8154758' 'IP_8171851' 'IP_8173459' 'IP_8191902' 'IP_8201833' 'IP_8213239' 'IP_8220633' 'IP_8224373' 'IP_8224538' 'IP_8226501' 'IP_8233849' 'IP_8235556' 'IP_8236536' 'IP_8236928' 'IP_8237561' 'IP_8246666' 'IP_8247808' 'IP_8252567' 'IP_8267064' 'IP_8273979' 'IP_8277853' 'IP_8280582' 'IP_8293383' 'IP_8295219' 'IP_8302048' 'IP_8313778' 'IP_8320831' 'IP_8326844' 'IP_8328099' 'IP_8329777' 'IP_8329991' 'IP_8335299' 'IP_8338213' 'IP_8340768' 'IP_8344324' 'IP_8349207' 'IP_8349964' 'IP_8352243' 'IP_8356528' 'IP_8367247' 'IP_8369251' 'IP_8374096' 'IP_8375053' 'IP_8375111' 'IP_8375879' 'IP_8385941' 'IP_8402958' 'IP_8412988' 'IP_8414061' 'IP_8418821' 'IP_8420428' 'IP_8427477' 'IP_8437394' 'IP_8439549' 'IP_8442277' 'IP_8446078' 'IP_8447624' 'IP_8474574' 'IP_8484959' 'IP_8490189' 'IP_8491618' 'IP_8494731' 'IP_8502723' 'IP_8506278' 'IP_8513373' 'IP_8522203' 'IP_8524154' 'IP_8526194' 'IP_8543718' 'IP_8547782' 'IP_8550051' 'IP_8550908' 'IP_8560544' 'IP_8561849' 'IP_8575171' 'IP_8590096' 'IP_8590724' 'IP_8593932' 'IP_8594522' 'IP_8599812' 'IP_8601697' 'IP_8614161' 'IP_8627268' 'IP_8627284' 'IP_8631184' 'IP_8631837' 'IP_8640021' 'IP_8642406' 'IP_8645527' 'IP_8650344' 'IP_8661408' 'IP_8663649' 'IP_8673016' 'IP_8678278' 'IP_8680583' 'IP_8691359' 'IP_8701813' 'IP_8702124' 'IP_8710422' 'IP_8715388' 'IP_8716491' 'IP_8717497' 'IP_8722561' 'IP_8723313' 'IP_8734172' 'IP_8742468' 'IP_8744388' 'IP_8753181' 'IP_8759634' 'IP_8766487' 'IP_8776152' 'IP_8785062' 'IP_8791409' 'IP_8793264' 'IP_8794667' 'IP_8803573' 'IP_8806573' 'IP_8829018' 'IP_8839187' 'IP_8851498' 'IP_8861178' 'IP_8869881' 'IP_8870268' 'IP_8872677' 'IP_8874484' 'IP_8915476' 'IP_8916001' 'IP_8916501' 'IP_8934738' 'IP_8936266' 'IP_8937782' 'IP_8940163' 'IP_8951606' 'IP_8964608' 'IP_8965673' 'IP_8968018' 'IP_8972853' 'IP_8974438' 'IP_8976307' 'IP_8980267' 'IP_8986063' 'IP_8988837' 'IP_8989787' 'IP_9000452' 'IP_9001826' 'IP_9002821' 'IP_9006739' 'IP_9013996' 'IP_9027799' 'IP_9029443' 'IP_9037179' 'IP_9042214' 'IP_9042531' 'IP_9042814' 'IP_9045497' 'IP_9051557' 'IP_9053722' 'IP_9054996' 'IP_9055249' 'IP_9056338' 'IP_9067016' 'IP_9067064' 'IP_9070827' 'IP_9086201' 'IP_9086717' 'IP_9087899' 'IP_9092661' 'IP_9102359' 'IP_9111321' 'IP_9115076' 'IP_9115451' 'IP_9122609' 'IP_9131123' 'IP_9137239' 'IP_9141207' 'IP_9144134' 'IP_9147454' 'IP_9153938' 'IP_9154808' 'IP_9160612' 'IP_9162256' 'IP_9164079' 'IP_9175987' 'IP_9180273' 'IP_9181133' 'IP_9183198' 'IP_9189624' 'IP_9190806' 'IP_9205477' 'IP_9205637' 'IP_9212266' 'IP_9218934' 'IP_9229871' 'IP_9232483' 'IP_9239044' 'IP_9245079' 'IP_9247426' 'IP_9248372' 'IP_9263039' 'IP_9268401' 'IP_9268661' 'IP_9272657' 'IP_9277763' 'IP_9284916' 'IP_9290101' 'IP_9297158' 'IP_9298839' 'IP_9299257' 'IP_9315526' 'IP_9315856' 'IP_9316163' 'IP_9328721' 'IP_9329623' 'IP_9335261' 'IP_9346829' 'IP_9348731' 'IP_9353814' 'IP_9355893' 'IP_9362467' 'IP_9370254' 'IP_9371649' 'IP_9373614' 'IP_9375901' 'IP_9392784' 'IP_9409608' 'IP_9416054' 'IP_9418573' 'IP_9422872' 'IP_9424914' 'IP_9433384' 'IP_9435698' 'IP_9438247' 'IP_9438537' 'IP_9450101' 'IP_9455054' 'IP_9460738' 'IP_9461058' 'IP_9494136' 'IP_9496389' 'IP_9505038' 'IP_9513376' 'IP_9536983' 'IP_9544996' 'IP_9551616' 'IP_9558959' 'IP_9563768' 'IP_9564988' 'IP_9576478' 'IP_9579751' 'IP_9583707' 'IP_9585016' 'IP_9594882' 'IP_9595383' 'IP_9612942' 'IP_9616517' 'IP_9619666' 'IP_9623174' 'IP_9624164' 'IP_9634846' 'IP_9635781' 'IP_9636064' 'IP_9639732' 'IP_9644384' 'IP_9644948' 'IP_9645497' 'IP_9649762' 'IP_9652966' 'IP_9661974' 'IP_9662709' 'IP_9663529' 'IP_9669483' 'IP_9672959' 'IP_9675066' 'IP_9679829' 'IP_9685257' 'IP_9689997' 'IP_9711347' 'IP_9712836' 'IP_9718058' 'IP_9718268' 'IP_9721483' 'IP_9736783' 'IP_9738076' 'IP_9746668' 'IP_9748593' 'IP_9751397' 'IP_9756233' 'IP_9756394' 'IP_9759699' 'IP_9770302' 'IP_9772004' 'IP_9786353' 'IP_9788106' 'IP_9802602' 'IP_9802994' 'IP_9806578' 'IP_9814841' 'IP_9818937' 'IP_9820186' 'IP_9824037' 'IP_9826072' 'IP_9826439' 'IP_9835231' 'IP_9835712' 'IP_9846692' 'IP_9849868' 'IP_9851472' 'IP_9853347' 'IP_9855344' 'IP_9856163' 'IP_9865971' 'IP_9888932' 'IP_9896679' 'IP_9898991' 'IP_9899071' 'IP_9901629' 'IP_9901906' 'IP_9902431' 'IP_9909432' 'IP_9921339' 'IP_9927968' 'IP_9935748' 'IP_9936643' 'IP_9942136' 'IP_9952683' 'IP_9954107' 'IP_9955501' 'IP_9960131' 'IP_9965542' 'IP_9989332' 'IP_9992027' 'IP_9996429']) 1 [])) (as.numeric (cols_py py_26_sid_96af 'age_approx')) 3 []))", 'session_id': '_sid_96af'}


## === cell 9
sample_submission = pd.read_csv(
    "../input/siim-isic-melanoma-classification/sample_submission.csv"
)



## === cell 10
test_pd = pd.read_csv(TEST_CSV, usecols=["image_name"])
sub = test_pd.copy()

if pred_p1 is not None and len(pred_p1) == len(sub):
    sub["target"] = pred_p1["target"].astype(float).to_numpy()
else:
    sub["target"] = 0.5

sample_submission = sample_submission[["image_name"]].merge(
    sub, on="image_name", how="left"
)

if sample_submission["target"].isna().any():
    missing = (
        sample_submission.loc[sample_submission["target"].isna(), "image_name"]
        .head(5)
        .tolist()
    )
    raise ValueError(f"Missing predictions for some image_name values, e.g.: {missing}")

sample_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2943840845.py in <cell line: 0>()
      2 sub = test_pd.copy()
      3 
----> 4 if pred_p1 is not None and len(pred_p1) == len(sub):
      5     sub["target"] = pred_p1["target"].astype(float).to_numpy()
      6 else:

NameError: name 'pred_p1' is not defined
