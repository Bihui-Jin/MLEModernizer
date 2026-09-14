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

3.7

# 2. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
seed = 197

import os

import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from packaging.version import Version
        import google.protobuf as _pb

        if Version(_pb.__version__) >= Version("5"):
            raise RuntimeError(f"Incompatible protobuf version: {_pb.__version__}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf") or m == "protobuf":
                sys.modules.pop(m, None)


_ensure_compatible_protobuf()

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "upb")

import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

random.seed(seed)
np.random.seed(seed)

tf.random.set_seed(seed)

print(tf.__version__)


## === cell 2
import pandas as pd

train = pd.read_csv('../input/train.tsv',  sep="\t")
test = pd.read_csv('../input/test.tsv',  sep="\t")


## === cell 3
train.head()


## === cell 4
train_df = train.sample(frac=1).reset_index(drop=True)

tf_estimator = tf.compat.v1.estimator

train_input_fn = tf_estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], num_epochs=None, shuffle=True
)

predict_train_input_fn = tf_estimator.inputs.pandas_input_fn(
    train_df, train_df["Sentiment"], shuffle=False
)


## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2665979120.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m [0;31m# Fix: 'tensorflow_estimator' is not installed in this environment.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;31m# Use the built-in Estimator API exposed via TF2 compat to preserve identical pandas_input_fn behavior.[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m [0mtf_estimator[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mcompat[0m[0;34m.[0m[0mv1[0m[0;34m.[0m[0mestimator[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m train_input_fn = tf_estimator.inputs.pandas_input_fn(

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/module_wrapper.py[0m in [0;36m_getattr[0;34m(self, name)[0m
[1;32m    230[0m     """
[1;32m    231[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 232[0;31m       [0mattr[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_tfmw_wrapped_module[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    233[0m     [0;32mexcept[0m [0mAttributeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    234[0m     [0;31m# Placeholder for Google-internal contrib error[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: module 'tensorflow.compat.v1' has no attribute 'estimator'

## === cell 5
embedded_text_feature_column = hub.text_embedding_column(
    key="Phrase", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim50/1")
