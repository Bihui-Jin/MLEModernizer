# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
)

import pandas as pd
import numpy as np

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub  # kept to preserve original imports/core logic
from keras import Sequential
from keras.layers import Dense
from scipy.stats import spearmanr

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
data = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
data.head()



## === cell 2
features = data[
    [
        "question_title",
        "question_body",
        "answer",
        "question_user_name",
        "answer_user_name",
    ]
]




## === cell 3
def fit_preprocessors(train_df):
    le = LabelEncoder()
    num_scaler = MinMaxScaler()

    cols = [
        "qa_id",
        "question_title",
        "question_body",
        "question_user_name",
        "question_user_page",
        "answer",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]

    mapper = DataFrameMapper(
        [
            ("qa_id", None),
            ("question_title", le),
            ("question_body", None),
            ("question_user_name", le),
            ("question_user_page", le),
            ("answer", None),
            ("answer_user_name", le),
            ("answer_user_page", le),
            ("url", le),
            ("category", le),
            ("host", le),
        ],
        df_out=True,
    )

    x_train_raw = pd.DataFrame(mapper.fit_transform(train_df), columns=cols)

    scale_cols = [
        "question_title",
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]
    num_scaler.fit(x_train_raw[scale_cols])

    return mapper, num_scaler


def transform_with_preprocessors(df, mapper, num_scaler):
    cols = [
        "qa_id",
        "question_title",
        "question_body",
        "question_user_name",
        "question_user_page",
        "answer",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]
    x = pd.DataFrame(mapper.transform(df), columns=cols)

    scale_cols = [
        "question_title",
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]
    scaled = pd.DataFrame(num_scaler.transform(x[scale_cols]), columns=scale_cols)

    x = x.drop(columns=scale_cols)
    x = pd.concat([x, scaled], axis=1)
    x = x.drop(columns="qa_id")
    return x




## === cell 4
def word2vec_fit(x_train_scaled):
    tfidf = TfidfVectorizer()

    mapper = DataFrameMapper(
        [
            ("question_title", None),
            ("question_body", tfidf),
            ("question_user_name", None),
            ("question_user_page", None),
            ("answer", tfidf),
            ("answer_user_name", None),
            ("answer_user_page", None),
            ("url", None),
            ("category", None),
            ("host", None),
        ]
    )

    vectors = mapper.fit(x_train_scaled)
    return vectors




## === cell 5
y = data[
    [
        "question_asker_intent_understanding",
        "question_body_critical",
        "question_conversational",
        "question_expect_short_answer",
        "question_fact_seeking",
        "question_has_commonly_accepted_answer",
        "question_interestingness_others",
        "question_interestingness_self",
        "question_multi_intent",
        "question_not_really_a_question",
        "question_opinion_seeking",
        "question_type_choice",
        "question_type_compare",
        "question_type_consequence",
        "question_type_definition",
        "question_type_entity",
        "question_type_instructions",
        "question_type_procedure",
        "question_type_reason_explanation",
        "question_type_spelling",
        "question_well_written",
        "answer_helpful",
        "answer_level_of_information",
        "answer_plausible",
        "answer_relevance",
        "answer_satisfaction",
        "answer_type_instructions",
        "answer_type_procedure",
        "answer_type_reason_explanation",
        "answer_well_written",
    ]
]



## === cell 6
def fit_preprocessors(train_df):
    from sklearn.preprocessing import OrdinalEncoder

    num_scaler = MinMaxScaler()
    cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    cols = [
        "qa_id",
        "question_title",
        "question_body",
        "question_user_name",
        "question_user_page",
        "answer",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]

    mapper = DataFrameMapper(
        [
            ("qa_id", None),
            ("question_title", cat_enc),
            ("question_body", None),
            ("question_user_name", cat_enc),
            ("question_user_page", cat_enc),
            ("answer", None),
            ("answer_user_name", cat_enc),
            ("answer_user_page", cat_enc),
            ("url", cat_enc),
            ("category", cat_enc),
            ("host", cat_enc),
        ],
        df_out=True,
    )

    x_train_raw = pd.DataFrame(mapper.fit_transform(train_df), columns=cols)

    scale_cols = [
        "question_title",
        "question_user_name",
        "question_user_page",
        "answer_user_name",
        "answer_user_page",
        "url",
        "category",
        "host",
    ]
    num_scaler.fit(x_train_raw[scale_cols])

    return mapper, num_scaler


data_train, data_valid, y_train, y_test = train_test_split(data, y, random_state=1)

enc_mapper, num_scaler = fit_preprocessors(data_train)

x_train_scaled = transform_with_preprocessors(data_train, enc_mapper, num_scaler)
x_test_scaled = transform_with_preprocessors(data_valid, enc_mapper, num_scaler)

word_vectors = word2vec_fit(x_train_scaled)

x_train = pd.DataFrame(word_vectors.transform(x_train_scaled))
x_test = pd.DataFrame(word_vectors.transform(x_test_scaled))


## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1894192776.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     58[0m [0mdata_train[0m[0;34m,[0m [0mdata_valid[0m[0;34m,[0m [0my_train[0m[0;34m,[0m [0my_test[0m [0;34m=[0m [0mtrain_test_split[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0my[0m[0;34m,[0m [0mrandom_state[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     59[0m [0;34m[0m[0m
[0;32m---> 60[0;31m [0menc_mapper[0m[0;34m,[0m [0mnum_scaler[0m [0;34m=[0m [0mfit_preprocessors[0m[0;34m([0m[0mdata_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     61[0m [0;34m[0m[0m
[1;32m     62[0m [0mx_train_scaled[0m [0;34m=[0m [0mtransform_with_preprocessors[0m[0;34m([0m[0mdata_train[0m[0;34m,[0m [0menc_mapper[0m[0;34m,[0m [0mnum_scaler[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1894192776.py[0m in [0;36mfit_preprocessors[0;34m(train_df)[0m
[1;32m     39[0m     )
[1;32m     40[0m [0;34m[0m[0m
[0;32m---> 41[0;31m     [0mx_train_raw[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mDataFrame[0m[0;34m([0m[0mmapper[0m[0;34m.[0m[0mfit_transform[0m[0;34m([0m[0mtrain_df[0m[0;34m)[0m[0;34m,[0m [0mcolumns[0m[0;34m=[0m[0mcols[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     42[0m [0;34m[0m[0m
[1;32m     43[0m     scale_cols = [

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py[0m in [0;36mwrapped[0;34m(self, X, *args, **kwargs)[0m
[1;32m    138[0m     [0;34m@[0m[0mwraps[0m[0;34m([0m[0mf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m     [0;32mdef[0m [0mwrapped[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 140[0;31m         [0mdata_to_wrap[0m [0;34m=[0m [0mf[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    141[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata_to_wrap[0m[0;34m,[0m [0mtuple[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    142[0m             [0;31m# only wrap the first output for cross decomposition[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn_pandas/dataframe_mapper.py[0m in [0;36mfit_transform[0;34m(self, X, y)[0m
[1;32m    441[0m         [0my[0m       [0mthe[0m [0mtarget[0m [0mvector[0m [0mrelative[0m [0mto[0m [0mX[0m[0;34m,[0m [0moptional[0m[0;34m[0m[0;34m[0m[0m
[1;32m    442[0m         """
[0;32m--> 443[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_transform[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/sklearn_pandas/dataframe_mapper.py[0m in [0;36m_transform[0;34m(self, X, y, do_fit)[0m
[1;32m    340[0m                     [0;32mif[0m [0mdo_fit[0m [0;32mand[0m [0mhasattr[0m[0;34m([0m[0mtransformers[0m[0;34m,[0m [0;34m'fit_transform'[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    341[0m                         [0mt1[0m [0;34m=[0m [0mdatetime[0m[0;34m.[0m[0mnow[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 342[0;31m                         [0mXt[0m [0;34m=[0m [0m_call_fit[0m[0;34m([0m[0mtransformers[0m[0;34m.[0m[0mfit_transform[0m[0;34m,[0m [0mXt[0m[0;34m,[0m [0my[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    343[0m                         [0mlogger[0m[0;34m.[0m[0minfo[0m[0;34m([0m[0;34mf"[FIT_TRANSFORM] {columns}: {_elapsed_secs(t1)} secs"[0m[0;34m)[0m  [0;31m# NOQA[0m[0;34m[0m[0;34m[0m[0m
[1;32m    344[0m                     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn_pandas/pipeline.py[0m in [0;36m_call_fit[0;34m(fit_method, X, y, **kwargs)[0m
[1;32m     22[0m     """
[1;32m     23[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 24[0;31m         [0;32mreturn[0m [0mfit_method[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     25[0m     [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m         [0;31m# fit takes only one argument[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py[0m in [0;36mwrapped[0;34m(self, X, *args, **kwargs)[0m
[1;32m    138[0m     [0;34m@[0m[0mwraps[0m[0;34m([0m[0mf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    139[0m     [0;32mdef[0m [0mwrapped[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 140[0;31m         [0mdata_to_wrap[0m [0;34m=[0m [0mf[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    141[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata_to_wrap[0m[0;34m,[0m [0mtuple[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    142[0m             [0;31m# only wrap the first output for cross decomposition[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36mfit_transform[0;34m(self, X, y, **fit_params)[0m
[1;32m    876[0m         [0;32mif[0m [0my[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    877[0m             [0;31m# fit method of arity 1 (unsupervised transformation)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 878[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0;34m**[0m[0mfit_params[0m[0;34m)[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mX[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    879[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    880[0m             [0;31m# fit method of arity 2 (supervised transformation)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_encoders.py[0m in [0;36mfit[0;34m(self, X, y)[0m
[1;32m   1301[0m [0;34m[0m[0m
[1;32m   1302[0m         [0;31m# `_fit` will only raise an error when `self.handle_unknown="error"`[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1303[0;31m         [0mself[0m[0;34m.[0m[0m_fit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mhandle_unknown[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mhandle_unknown[0m[0;34m,[0m [0mforce_all_finite[0m[0;34m=[0m[0;34m"allow-nan"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1304[0m [0;34m[0m[0m
[1;32m   1305[0m         [0mcardinalities[0m [0;34m=[0m [0;34m[[0m[0mlen[0m[0;34m([0m[0mcategories[0m[0;34m)[0m [0;32mfor[0m [0mcategories[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mcategories_[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_encoders.py[0m in [0;36m_fit[0;34m(self, X, handle_unknown, force_all_finite, return_counts)[0m
[1;32m     72[0m         [0mself[0m[0;34m.[0m[0m_check_n_features[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mreset[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     73[0m         [0mself[0m[0;34m.[0m[0m_check_feature_names[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mreset[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 74[0;31m         X_list, n_samples, n_features = self._check_X(
[0m[1;32m     75[0m             [0mX[0m[0;34m,[0m [0mforce_all_finite[0m[0;34m=[0m[0mforce_all_finite[0m[0;34m[0m[0;34m[0m[0m
[1;32m     76[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_encoders.py[0m in [0;36m_check_X[0;34m(self, X, force_all_finite)[0m
[1;32m     44[0m         [0;32mif[0m [0;32mnot[0m [0;34m([0m[0mhasattr[0m[0;34m([0m[0mX[0m[0;34m,[0m [0;34m"iloc"[0m[0;34m)[0m [0;32mand[0m [0mgetattr[0m[0;34m([0m[0mX[0m[0;34m,[0m [0;34m"ndim"[0m[0;34m,[0m [0;36m0[0m[0;34m)[0m [0;34m==[0m [0;36m2[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     45[0m             [0;31m# if not a dataframe, do normal check_array validation[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 46[0;31m             [0mX_temp[0m [0;34m=[0m [0mcheck_array[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0mforce_all_finite[0m[0;34m=[0m[0mforce_all_finite[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     47[0m             [0;32mif[0m [0;32mnot[0m [0mhasattr[0m[0;34m([0m[0mX[0m[0;34m,[0m [0;34m"dtype"[0m[0;34m)[0m [0;32mand[0m [0mnp[0m[0;34m.[0m[0missubdtype[0m[0;34m([0m[0mX_temp[0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mstr_[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m                 [0mX[0m [0;34m=[0m [0mcheck_array[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mobject[0m[0;34m,[0m [0mforce_all_finite[0m[0;34m=[0m[0mforce_all_finite[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py[0m in [0;36mcheck_array[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)[0m
[1;32m    900[0m             [0;31m# If input is 1D raise error[0m[0;34m[0m[0;34m[0m[0m
[1;32m    901[0m             [0;32mif[0m [0marray[0m[0;34m.[0m[0mndim[0m [0;34m==[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 902[0;31m                 raise ValueError(
[0m[1;32m    903[0m                     [0;34m"Expected 2D array, got 1D array instead:\narray={}.\n"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    904[0m                     [0;34m"Reshape your data either using array.reshape(-1, 1) if "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: question_title: Expected 2D array, got 1D array instead:
array=['How are pixels actually shown on display'
 'Can you send data usefully over one wire, literally one wire?'
 'Looking for advanced GeoServer tutorials' ...
 'Cannot access non-static field'
 'Is a password easier to brute force if it contains a repeating pattern?'
 'Using nofollow when crosslinking my own sites'].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

## === cell 7
x_train.head()
