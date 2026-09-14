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
The training data is the purchase history of customers across time. The task is to predict what articles each customer will purchase in the 7-day period immediately after the training data ends.

## Metric
Mean Average Precision @ 12 (MAP@12):

$$
\text{MAP@12}=\frac{1}{U} \sum_{u=1}^U \frac{1}{\min (m, 12)} \sum_{k=1}^{\min (n, 12)} P(k) \times \text{rel}(k)
$$

where $U$ is the number of customers, $P(k)$ is the precision at cutoff $k, n$ is the number predictions per customer, $m$ is the number of ground truth values per customer, and $\text{rel}(k)$ is an indicator function equaling 1 if the item at rank $k$ is a relevant (correct) label, zero otherwise.

You must make predictions for all `customer_id` values found in the sample submission. All customers who made purchases during the test period are scored, regardless of whether they had purchase history in the training data.

## Submission Format
For each `customer_id` observed in the training data, you may predict up to 12 labels for the `article_id`, which is the predicted items a customer will buy in the next 7-day period after the training time period. The file should contain a header and have the following format:

```
customer_id,prediction
00000dba,0706016001 0706016002 0372860001 ...
0000423b,0706016001 0706016002 0372860001 ...
...
```

## Dataset
- **images/** - a folder of images corresponding to each `article_id`; images are placed in subfolders starting with the first three digits of the `article_id`; note, not all `article_id` values have a corresponding image.
- **articles.csv** - detailed metadata for each `article_id` available for purchase
- **customers.csv** - metadata for each `customer_id` in dataset
- **sample_submission.csv** - a sample submission file in the correct format
- **transactions_train.csv** - the training data, consisting of the purchases each customer for each date, as well as additional information. Duplicate rows correspond to multiple purchases of the same item. Your task is to predict the `article_id`s each customer will purchase during the 7-day period immediately after the training data period.

# 2. Python version

3.14

# 3. Installed packages

cuml-cu12==25.2.1
geopandas==0.14.4
libcuml-cu12==25.2.1
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
            articles.csv (105543 lines)
            articles.csv.zip (4.4 MB)
            customers.csv (1371981 lines)
            customers.csv.zip (102.4 MB)
            description.md (74 lines)
            images.zip (30.0 GB)
            sample_submission.csv (1371981 lines)
            sample_submission.csv.zip (53.3 MB)
            transactions_train.csv (31521961 lines)
            transactions_train.csv.zip (604.1 MB)
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
            images/
                010/
                    0108775015.jpg (154.6 kB)
                    0108775044.jpg (106.7 kB)
                    ... and 1 other files
                011/
                    0110065001.jpg (148.6 kB)
                    0110065002.jpg (85.7 kB)
                    ... and 18 other files
                ... and 84 other folders
        input/
            articles.csv (105543 lines)
            articles.csv.zip (4.4 MB)
            customers.csv (1371981 lines)
            customers.csv.zip (102.4 MB)
            description.md (74 lines)
            images.zip (30.0 GB)
            sample_submission.csv (1371981 lines)
            sample_submission.csv.zip (53.3 MB)
            transactions_train.csv (31521961 lines)
            transactions_train.csv.zip (604.1 MB)
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
            images/
                010/
                    0108775015.jpg (154.6 kB)
                    0108775044.jpg (106.7 kB)
                    ... and 1 other files
                011/
                    0110065001.jpg (148.6 kB)
                    0110065002.jpg (85.7 kB)
                    ... and 18 other files
                ... and 84 other folders
        working/
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
```

-> data/articles.csv has 105542 rows and 25 columns.
The columns are: article_id, product_code, prod_name, product_type_no, product_type_name, product_group_name, graphical_appearance_no, graphical_appearance_name, colour_group_code, colour_group_name, perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name, department_no... and 10 more columns

-> data/customers.csv has 1371980 rows and 7 columns.
The columns are: customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code

-> data/h-and-m-personalized-fashion-recommendations/articles.csv has 105542 rows and 25 columns.
The columns are: article_id, product_code, prod_name, product_type_no, product_type_name, product_group_name, graphical_appearance_no, graphical_appearance_name, colour_group_code, colour_group_name, perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name, department_no... and 10 more columns

-> data/h-and-m-personalized-fashion-recommendations/customers.csv has 1371980 rows and 7 columns.
The columns are: customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code

-> data/h-and-m-personalized-fashion-recommendations/sample_submission.csv has 1371980 rows and 2 columns.
The columns are: customer_id, prediction

-> data/h-and-m-personalized-fashion-recommendations/transactions_train.csv has 31521960 rows and 5 columns.
The columns are: t_dat, customer_id, article_id, price, sales_channel_id

-> data/sample_submission.csv has 1371980 rows and 2 columns.
The columns are: customer_id, prediction

-> data/transactions_train.csv has 31521960 rows and 5 columns.
The columns are: t_dat, customer_id, article_id, price, sales_channel_id

-> (stopped after 10 files for performance)

# 5. Target score

0.028964833733073

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%%time
import os
import sys
import copy
from datetime import datetime
import gc
import pickle as pkl
import shelve

import pandas as pd
import numpy as np
import cudf
    
sys.path.append("../input/")
from helper import io as h_io, sub as h_sub, cv as h_cv, fe as h_fe
from helper import modeling as h_modeling, candidates as h_can, pairs as h_pairs

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
CUDARuntimeError                          Traceback (most recent call last)
<timed exec> in <module>

/usr/local/lib/python3.11/dist-packages/cudf/__init__.py in <module>
     18 
     19 _setup_numba()
---> 20 validate_setup()
     21 
     22 import cupy

/usr/local/lib/python3.11/dist-packages/cudf/utils/gpu_utils.py in validate_setup()
     53     except CUDARuntimeError as e:
     54         if e.status in notify_caller_errors:
---> 55             raise e
     56         # If there is no GPU detected, set `gpus_count` to -1
     57         gpus_count = -1

/usr/local/lib/python3.11/dist-packages/cudf/utils/gpu_utils.py in validate_setup()
     50 
     51     try:
---> 52         gpus_count = getDeviceCount()
     53     except CUDARuntimeError as e:
     54         if e.status in notify_caller_errors:

/usr/local/lib/python3.11/dist-packages/rmm/_cuda/gpu.py in getDeviceCount()
    100     status, count = runtime.cudaGetDeviceCount()
    101     if status != runtime.cudaError_t.cudaSuccess:
--> 102         raise CUDARuntimeError(status)
    103     return count
    104 

CUDARuntimeError: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version

## === cell 1
from datetime import timedelta

def day_week_numbers_fixed(dates):
    """
    Fix for: Series has no attribute applymap
    Keeps the same logic as helper/fe.py but uses vectorized ceil-division.
    Works with cudf (GPU) and falls back to pandas if cudf path fails.
    """
    try:
        import cudf

        pd_dates = cudf.to_datetime(dates)
        unique_dates = cudf.Series(pd_dates.unique())

        numbered_days = unique_dates - unique_dates.min() + timedelta(1)
        numbered_days = numbered_days.dt.days

        extra_days = int(numbered_days.max() % 7)
        adjusted_days = (numbered_days - extra_days).astype("int32")

        day_weeks = -((-adjusted_days) // 7)

        day_weeks_map = (
            cudf.DataFrame({"day_weeks": day_weeks, "unique_dates": unique_dates})
            .set_index("unique_dates")["day_weeks"]
        )

        all_day_weeks = pd_dates.map(day_weeks_map).astype("int8")
        return all_day_weeks

    except Exception:
        import pandas as pd

        pd_dates = pd.to_datetime(dates)
        unique_dates = pd.Series(pd_dates.unique())

        numbered_days = unique_dates - unique_dates.min() + timedelta(1)
        numbered_days = numbered_days.dt.days

        extra_days = int(numbered_days.max() % 7)
        adjusted_days = (numbered_days - extra_days).astype("int32")

        day_weeks = -((-adjusted_days) // 7)

        day_weeks_map = pd.Series(day_weeks.values, index=unique_dates.values)
        return pd_dates.map(day_weeks_map).astype("int8")

h_fe.day_week_numbers = day_week_numbers_fixed


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2451674918.py in <cell line: 0>()
     48 
     49 # patch vào module helper.fe đã import
---> 50 h_fe.day_week_numbers = day_week_numbers_fixed

NameError: name 'h_fe' is not defined

## === cell 2
%%time

c, t, a = h_io.load_data(files=['customers.csv', 'transactions_train.csv', 'articles.csv'])        

index_to_id_dict_path = h_fe.reduce_customer_id_memory(c, [t])
t["week_number"] = h_fe.day_week_numbers(t["t_dat"])
t["t_dat"] = h_fe.day_numbers(t["t_dat"])

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'h_io' is not defined

## === cell 3
%%time

pairs_per_item = 5

week_number_pairs = {}
for week_number in [96, 97, 98, 99, 100, 101, 102, 103, 104]:
    print(f"Creating pairs for week number {week_number}")
    week_number_pairs[week_number] = h_pairs.create_pairs(
        t, week_number, pairs_per_item, verbose=False
    )

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'h_pairs' is not defined

## === cell 4
def create_candidates_with_features_df(t, c, a, customer_batch=None, **kwargs):
    features_df, label_df = h_cv.feature_label_split(
        t, kwargs["label_week"], kwargs["feature_periods"]
    )
    
    features_df["t_dat"] = h_fe.how_many_ago(features_df["t_dat"])
    features_df["week_number"] = h_fe.how_many_ago(features_df["week_number"])
    
    article_pairs_df = week_number_pairs[kwargs["label_week"]-1]
    
    if len(label_df) > 0:
        customers = label_df["customer_id"].unique()
    elif customer_batch is not None:
        customers = customer_batch
    else:
        customers = None
    
    
    features_db = shelve.open("features_db") 
    
    recent_customer_cand, features_db["customer_article"] = (
        h_can.create_recent_customer_candidates(
            features_df,
            kwargs["ca_num_weeks"],
            customers=customers,
        )
    )
    
    (cust_last_week_cand,
     cust_last_week_pair_cand,
     features_db["clw"],
     features_db["clw_pairs"]) = h_can.create_last_customer_weeks_and_pairs(
        features_df,
        article_pairs_df,
        kwargs["clw_num_weeks"],
        kwargs["clw_num_pair_weeks"],
        customers=customers,
    )
    
    _, features_db["popular_articles"] = h_can.create_popular_article_cand(
        features_df,
        c,
        a,
        kwargs["pa_num_weeks"],
        kwargs["hier_col"],
        num_candidates=kwargs["num_recent_candidates"],
        num_articles=kwargs["num_recent_articles"],
        customers=customers,
    )
    age_bucket_can, _, _ = h_can.create_age_bucket_candidates(
        features_df,
        c,
        kwargs["num_age_buckets"],
        articles=kwargs["num_recent_articles"],
        customers=customers,
    )
    
    cand = [recent_customer_cand, cust_last_week_cand, cust_last_week_pair_cand, age_bucket_can]
    cand = cudf.concat(cand).drop_duplicates()
    cand = cand.sort_values(["customer_id", "article_id"]).reset_index(drop=True)
    
    del recent_customer_cand, cust_last_week_cand, cust_last_week_pair_cand, age_bucket_can
    
    cand = h_can.filter_candidates(cand, t, **kwargs)
    
    h_fe.create_cust_hier_features(features_df, a, kwargs["hier_cols"], features_db)
    h_fe.create_price_features(features_df, features_db)
    h_fe.create_cust_features(c, features_db)
    h_fe.create_article_cust_features(features_df, c, features_db)
    h_fe.create_lag_features(features_df, a, kwargs["lag_days"], features_db)
    h_fe.create_rebuy_features(features_df, features_db)
    h_fe.create_cust_t_features(features_df, a, features_db)
    h_fe.create_art_t_features(features_df, features_db)
    
    del features_df

    if customers is not None:
        cand = cand[cand["customer_id"].isin(customers)]
    
    if kwargs["cv"]:
        ground_truth_candidates = label_df[["customer_id", "article_id"]].drop_duplicates()
        h_cv.report_candidates(cand, ground_truth_candidates)
        del ground_truth_candidates        
    
    cand_with_f_df = h_can.add_features_to_candidates(
        cand, features_db, c, a
    )
    
    for article_col in kwargs["article_columns"]:
        art_col_map = a.set_index("article_id")[article_col]
        cand_with_f_df[article_col] = cand_with_f_df["article_id"].map(art_col_map)
    
    if kwargs["selected_features"] is not None:
        cand_with_f_df = cand_with_f_df[
            ["customer_id", "article_id"] + kwargs["selected_features"]
        ]
        
    features_db.close()
    os.remove("features_db.bak"), os.remove("features_db.dir"), os.remove("features_db.dat")
    
    assert len(cand) == len(cand_with_f_df), "seem to have duplicates in the feature dfs"
    del cand
    
    return cand_with_f_df, label_df

## === cell 5
def calculate_model_score(ids_df, preds, truth_df):
    predictions = h_modeling.create_predictions(ids_df, preds)
    true_labels = h_cv.ground_truth(truth_df).set_index("customer_id")["prediction"]
    score = round(h_cv.comp_average_precision(true_labels, predictions),5)
    
    return score

## === cell 6
cv_params = {
    "cv": True,
    "feature_periods": 105,
    "label_week": 104,
    "index_to_id_dict_path": index_to_id_dict_path,
    "pairs_file_version": "_v3_5_ex",
    "num_recent_candidates": 36,
    "num_recent_articles": 12,
    "hier_col": "department_no",
    "ca_num_weeks": 3,
    "clw_num_weeks": 12,
    "clw_num_pair_weeks": 2,
    "pa_num_weeks": 1,
    "num_age_buckets": 4,
    "filter_recent_art_weeks": 1,
    "filter_num_articles": None,
    "lag_days": [1, 3, 14, 30],
    "article_columns": ["index_code"],
    "hier_cols": [
        "department_no", "section_no", "index_group_no", "index_code",
        "product_type_no", "product_group_name"
    ],
    "selected_features": None,
    "lgbm_params": {"n_estimators": 200, "num_leaves": 20},
    "log_evaluation": 10,
    "early_stopping": 20,
    "eval_at": 12,
    "save_model": True,
    "num_concats": 5,
}
sub_params = {
    "cv": False,
    "feature_periods": 105,
    "label_week": 105,
    "index_to_id_dict_path": index_to_id_dict_path,
    "pairs_file_version": "_v3_5_ex",
    "num_recent_candidates": 60,
    "num_recent_articles": 12,
    "hier_col": "department_no",
    "ca_num_weeks": 3,
    "clw_num_weeks": 12,
    "clw_num_pair_weeks": 2,
    "pa_num_weeks": 1,
    "num_age_buckets": 4,
    "filter_recent_art_weeks": 1,
    "filter_num_articles": None,
    "lag_days": [1, 3, 14, 30],
    "article_columns": ["index_code"],
    "hier_cols": [
        "department_no", "section_no", "index_group_no", "index_code",
        "product_type_no", "product_group_name"
    ],
    "selected_features": None,
    "lgbm_params": {
        "n_estimators": 150,
        "num_leaves": 20,    
    },
    "log_evaluation": 10,
    "eval_at": 12,
    "prediction_models": ["model_104", "model_105"],
    "save_model": True,
    "num_concats": 5,
}

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1421280664.py in <cell line: 0>()
      3     "feature_periods": 105,
      4     "label_week": 104,
----> 5     "index_to_id_dict_path": index_to_id_dict_path,
      6     "pairs_file_version": "_v3_5_ex",
      7     "num_recent_candidates": 36,

NameError: name 'index_to_id_dict_path' is not defined

## === cell 7
cand_features_func = create_candidates_with_features_df

scoring_func = calculate_model_score

## === cell 8
import os as _os

_real_remove = _os.remove

def _safe_remove(path):
    try:
        _real_remove(path)
    except FileNotFoundError:
        if str(path).startswith("features_db"):
            return
        raise

_os.remove = _safe_remove


## === cell 9
import pandas as pd
import numpy as np

def _encode_pandas_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype("int8")
    if pd.api.types.is_categorical_dtype(s):
        return s.cat.codes.astype("int32")
    if pd.api.types.is_object_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() >= 0.9:
            return num
        codes, _ = pd.factorize(s, sort=True)
        return pd.Series(codes, index=s.index).astype("int32")
    if np.issubdtype(s.dtype, np.datetime64):
        return s.view("int64")
    return s

def _encode_cudf_series(s):
    dt = s.dtype
    dt_str = str(dt)
    if dt_str == "category":
        return s.cat.codes.astype("int32")
    if dt_str in ("object", "str") or "str" in dt_str:
        codes, _ = s.factorize()
        return codes.astype("int32")
    if "datetime64" in dt_str:
        return s.astype("int64")
    if dt_str.startswith("bool"):
        return s.astype("int8")
    return s

def prep_cudf_to_pandas_patched(df, inplace=True):
    out = df if inplace else (df.copy(deep=True) if hasattr(df, "copy") else df)

    try:
        import cudf
        if isinstance(out, cudf.DataFrame):
            for col in out.columns:
                out[col] = _encode_cudf_series(out[col])
            return out
        if isinstance(out, cudf.Series):
            return _encode_cudf_series(out)
    except Exception:
        pass

    if isinstance(out, pd.DataFrame):
        for col in out.columns:
            out[col] = _encode_pandas_series(out[col])
        return out
    if isinstance(out, pd.Series):
        return _encode_pandas_series(out)

    return out

h_modeling.prep_cudf_to_pandas = prep_cudf_to_pandas_patched


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4109897976.py in <cell line: 0>()
     60 
     61 # monkey-patch vào helper.modeling
---> 62 h_modeling.prep_cudf_to_pandas = prep_cudf_to_pandas_patched

NameError: name 'h_modeling' is not defined

## === cell 10
%%time
cv_weeks = [104]
results = h_modeling.run_all_cvs(
    t, c, a, cand_features_func, scoring_func, 
    cv_weeks=cv_weeks, **cv_params
)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'h_modeling' is not defined

## === cell 11
from cuml.fil import ForestInference as _FI

_real_load = _FI.load

def _load_compat(*args, output_class=None, is_classifier=None, **kwargs):
    if is_classifier is None and output_class is not None:
        is_classifier = output_class
    return _real_load(*args, is_classifier=is_classifier, **kwargs)

_FI.load = staticmethod(_load_compat)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
CUDARuntimeError                          Traceback (most recent call last)
/tmp/ipykernel_11/2770241157.py in <cell line: 0>()
----> 1 from cuml.fil import ForestInference as _FI
      2 
      3 _real_load = _FI.load
      4 
      5 def _load_compat(*args, output_class=None, is_classifier=None, **kwargs):

/usr/local/lib/python3.11/dist-packages/cuml/__init__.py in <module>
     25     del libcuml
     26 
---> 27 from cuml.internals.base import Base, UniversalBase
     28 from cuml.internals.available_devices import is_cuda_available
     29 

/usr/local/lib/python3.11/dist-packages/cuml/internals/__init__.py in <module>
     16 
     17 from cuml.internals.available_devices import is_cuda_available
---> 18 from cuml.internals.base_helpers import BaseMetaClass, _tags_class_and_instance
     19 from cuml.internals.api_decorators import (
     20     _deprecate_pos_args,

/usr/local/lib/python3.11/dist-packages/cuml/internals/base_helpers.py in <module>
     18 import typing
     19 
---> 20 from cuml.internals.api_decorators import (
     21     api_base_return_generic,
     22     api_base_return_array,

/usr/local/lib/python3.11/dist-packages/cuml/internals/api_decorators.py in <module>
     22 
     23 # TODO: Try to resolve circular import that makes this necessary:
---> 24 from cuml.internals import input_utils as iu
     25 from cuml.internals.api_context_managers import BaseReturnAnyCM
     26 from cuml.internals.api_context_managers import BaseReturnArrayCM

/usr/local/lib/python3.11/dist-packages/cuml/internals/input_utils.py in <module>
     18 from typing import Literal
     19 
---> 20 from cuml.internals.array import CumlArray
     21 from cuml.internals.array_sparse import SparseCumlArray
     22 from cuml.internals.global_settings import GlobalSettings

/usr/local/lib/python3.11/dist-packages/cuml/internals/array.py in <module>
     19 import pickle
     20 
---> 21 from cuml.internals.global_settings import GlobalSettings
     22 from cuml.internals.logger import debug
     23 from cuml.internals.mem_type import MemoryType, MemoryTypeError

/usr/local/lib/python3.11/dist-packages/cuml/internals/global_settings.py in <module>
     18 import threading
     19 from cuml.internals.available_devices import is_cuda_available
---> 20 from cuml.internals.device_type import DeviceType
     21 from cuml.internals.mem_type import MemoryType
     22 from cuml.internals.safe_imports import cpu_only_import, gpu_only_import

/usr/local/lib/python3.11/dist-packages/cuml/internals/device_type.py in <module>
     17 
     18 from enum import Enum, auto
---> 19 from cuml.internals.mem_type import MemoryType
     20 
     21 

/usr/local/lib/python3.11/dist-packages/cuml/internals/mem_type.py in <module>
     20 from cuml.internals.safe_imports import cpu_only_import, gpu_only_import
     21 
---> 22 cudf = gpu_only_import("cudf")
     23 cp = gpu_only_import("cupy")
     24 cpx_sparse = gpu_only_import("cupyx.scipy.sparse")

/usr/local/lib/python3.11/dist-packages/cuml/internals/safe_imports.py in gpu_only_import(module, alt)
    360     """
    361     if GPU_ENABLED:
--> 362         return importlib.import_module(module)
    363     else:
    364         return safe_import(

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/local/lib/python3.11/dist-packages/cudf/__init__.py in <module>
     18 
     19 _setup_numba()
---> 20 validate_setup()
     21 
     22 import cupy

/usr/local/lib/python3.11/dist-packages/cudf/utils/gpu_utils.py in validate_setup()
     53     except CUDARuntimeError as e:
     54         if e.status in notify_caller_errors:
---> 55             raise e
     56         # If there is no GPU detected, set `gpus_count` to -1
     57         gpus_count = -1

/usr/local/lib/python3.11/dist-packages/cudf/utils/gpu_utils.py in validate_setup()
     50 
     51     try:
---> 52         gpus_count = getDeviceCount()
     53     except CUDARuntimeError as e:
     54         if e.status in notify_caller_errors:

/usr/local/lib/python3.11/dist-packages/rmm/_cuda/gpu.py in getDeviceCount()
    100     status, count = runtime.cudaGetDeviceCount()
    101     if status != runtime.cudaError_t.cudaSuccess:
--> 102         raise CUDARuntimeError(status)
    103     return count
    104 

CUDARuntimeError: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version

## === cell 12
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Parameter `output_class` was deprecated.*",
    category=FutureWarning,
)


## === cell 13
%%time
gc.collect()
h_modeling.full_sub_train_run(t, c, a, cand_features_func, scoring_func, **sub_params)
predictions = h_modeling.full_sub_predict_run(
    t, c, a, cand_features_func, **sub_params
)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'h_modeling' is not defined

## === cell 14
sub = h_sub.create_sub(c["customer_id"], predictions, index_to_id_dict_path)
sub.to_csv('dev_submission.csv', index=False)

display(sub.head())
print(sub.shape)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/762034876.py in <cell line: 0>()
----> 1 sub = h_sub.create_sub(c["customer_id"], predictions, index_to_id_dict_path)
      2 sub.to_csv('dev_submission.csv', index=False)
      3 
      4 display(sub.head())
      5 print(sub.shape)

NameError: name 'h_sub' is not defined
