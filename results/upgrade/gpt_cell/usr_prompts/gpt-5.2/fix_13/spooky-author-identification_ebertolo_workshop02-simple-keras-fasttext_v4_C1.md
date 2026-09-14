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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os

import sys
import subprocess

try:
    from google.protobuf.message_factory import MessageFactory

    _ = MessageFactory().GetPrototype  # noqa: B018
except Exception:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf<4"]
    )

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np
import pandas as pd
from collections import defaultdict

import tf_keras as keras
from tf_keras.layers import Dense, GlobalAveragePooling1D, Embedding
from tf_keras.callbacks import EarlyStopping
from tf_keras.models import Sequential
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.utils import to_categorical

from sklearn.model_selection import train_test_split

np.random.seed(7)


## --- ERROR in cell 0, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1067619058.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     23[0m [0;32mfrom[0m [0mcollections[0m [0;32mimport[0m [0mdefaultdict[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;34m[0m[0m
[0;32m---> 25[0;31m [0;32mimport[0m [0mtf_keras[0m [0;32mas[0m [0mkeras[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     26[0m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0mlayers[0m [0;32mimport[0m [0mDense[0m[0;34m,[0m [0mGlobalAveragePooling1D[0m[0;34m,[0m [0mEmbedding[0m[0;34m[0m[0;34m[0m[0m
[1;32m     27[0m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0mcallbacks[0m [0;32mimport[0m [0mEarlyStopping[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tf_keras/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;34m"""AUTOGENERATED. DO NOT EDIT."""[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0;32mfrom[0m [0mtf_keras[0m [0;32mimport[0m [0m__internal__[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfrom[0m [0mtf_keras[0m [0;32mimport[0m [0mactivations[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mtf_keras[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tf_keras/__internal__/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;34m"""AUTOGENERATED. DO NOT EDIT."""[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mlayers[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mlosses[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tf_keras/__internal__/backend/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;34m"""AUTOGENERATED. DO NOT EDIT."""[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m [0;32mimport[0m [0m_initialize_variables[0m [0;32mas[0m [0minitialize_variables[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m [0;32mimport[0m [0mtrack_variable[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tf_keras/src/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     19[0m """
[1;32m     20[0m [0;34m[0m[0m
[0;32m---> 21[0;31m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mapplications[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mdistribute[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mlayers[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tf_keras/src/applications/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     16[0m [0;34m[0m[0m
[1;32m     17[0m [0;34m[0m[0m
[0;32m---> 18[0;31m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapplications[0m[0;34m.[0m[0mconvnext[0m [0;32mimport[0m [0mConvNeXtBase[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapplications[0m[0;34m.[0m[0mconvnext[0m [0;32mimport[0m [0mConvNeXtLarge[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mapplications[0m[0;34m.[0m[0mconvnext[0m [0;32mimport[0m [0mConvNeXtSmall[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tf_keras/src/applications/convnext.py[0m in [0;36m<module>[0;34m[0m
[1;32m     24[0m [0;34m[0m[0m
[1;32m     25[0m [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 26[0;31m [0;32mimport[0m [0mtensorflow[0m[0;34m.[0m[0mcompat[0m[0;34m.[0m[0mv2[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     27[0m [0;34m[0m[0m
[1;32m     28[0m [0;32mfrom[0m [0mtf_keras[0m[0;34m.[0m[0msrc[0m [0;32mimport[0m [0mbackend[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     47[0m [0m_tf2[0m[0;34m.[0m[0menable[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m [0;34m[0m[0m
[0;32m---> 49[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__internal__[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0m__operators__[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m [0;32mimport[0m [0maudio[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mautograph[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdecorator[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0m_api[0m[0;34m.[0m[0mv2[0m[0;34m.[0m[0m__internal__[0m [0;32mimport[0m [0mdispatch[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mimport[0m [0msys[0m [0;32mas[0m [0m_sys[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mag_ctx[0m [0;32mimport[0m [0mcontrol_status_ctx[0m [0;31m# line: 34[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mimpl[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mtf_convert[0m [0;31m# line: 493[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py[0m in [0;36m<module>[0;34m[0m
[1;32m     19[0m [0;32mimport[0m [0mthreading[0m[0;34m[0m[0;34m[0m[0m
[1;32m     20[0m [0;34m[0m[0m
[0;32m---> 21[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mag_logging[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     22[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mutil[0m[0;34m.[0m[0mtf_export[0m [0;32mimport[0m [0mtf_export[0m[0;34m[0m[0;34m[0m[0m
[1;32m     23[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     15[0m [0;34m"""Utility module that contains APIs usable in the generated code."""[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;34m[0m[0m
[0;32m---> 17[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mcontext_managers[0m [0;32mimport[0m [0mcontrol_dependency_on_returns[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmisc[0m [0;32mimport[0m [0malias_tensors[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mautograph[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mtensor_list[0m [0;32mimport[0m [0mdynamic_list_append[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py[0m in [0;36m<module>[0;34m[0m
[1;32m     17[0m [0;32mimport[0m [0mcontextlib[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;34m[0m[0m
[0;32m---> 19[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mops[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mpython[0m[0;34m.[0m[0mops[0m [0;32mimport[0m [0mtensor_array_ops[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36m<module>[0;34m[0m
[1;32m     31[0m [0;34m[0m[0m
[1;32m     32[0m [0;32mfrom[0m [0mgoogle[0m[0;34m.[0m[0mprotobuf[0m [0;32mimport[0m [0mmessage[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 33[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mattr_value_pb2[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfull_type_pb2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mfunction_pb2[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py[0m in [0;36m<module>[0;34m[0m
[1;32m     12[0m [0;34m[0m[0m
[1;32m     13[0m [0;34m[0m[0m
[0;32m---> 14[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mtensor_pb2[0m [0;32mas[0m [0mtensorflow_dot_core_dot_framework_dot_tensor__pb2[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     15[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mtensor_shape_pb2[0m [0;32mas[0m [0mtensorflow_dot_core_dot_framework_dot_tensor__shape__pb2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mtypes_pb2[0m [0;32mas[0m [0mtensorflow_dot_core_dot_framework_dot_types__pb2[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/tensor_pb2.py[0m in [0;36m<module>[0;34m[0m
[1;32m     12[0m [0;34m[0m[0m
[1;32m     13[0m [0;34m[0m[0m
[0;32m---> 14[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mresource_handle_pb2[0m [0;32mas[0m [0mtensorflow_dot_core_dot_framework_dot_resource__handle__pb2[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     15[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mtensor_shape_pb2[0m [0;32mas[0m [0mtensorflow_dot_core_dot_framework_dot_tensor__shape__pb2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mtypes_pb2[0m [0;32mas[0m [0mtensorflow_dot_core_dot_framework_dot_types__pb2[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/resource_handle_pb2.py[0m in [0;36m<module>[0;34m[0m
[1;32m     12[0m [0;34m[0m[0m
[1;32m     13[0m [0;34m[0m[0m
[0;32m---> 14[0;31m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mtensor_shape_pb2[0m [0;32mas[0m [0mtensorflow_dot_core_dot_framework_dot_tensor__shape__pb2[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     15[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mframework[0m [0;32mimport[0m [0mtypes_pb2[0m [0;32mas[0m [0mtensorflow_dot_core_dot_framework_dot_types__pb2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/tensor_shape_pb2.py[0m in [0;36m<module>[0;34m[0m
[1;32m     17[0m [0;34m[0m[0m
[1;32m     18[0m [0m_builder[0m[0;34m.[0m[0mBuildMessageAndEnumDescriptors[0m[0;34m([0m[0mDESCRIPTOR[0m[0;34m,[0m [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 19[0;31m [0m_builder[0m[0;34m.[0m[0mBuildTopDescriptorsAndMessages[0m[0;34m([0m[0mDESCRIPTOR[0m[0;34m,[0m [0;34m'tensorflow.core.framework.tensor_shape_pb2'[0m[0;34m,[0m [0mglobals[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;32mif[0m [0m_descriptor[0m[0;34m.[0m[0m_USE_C_DESCRIPTORS[0m [0;34m==[0m [0;32mFalse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py[0m in [0;36mBuildTopDescriptorsAndMessages[0;34m(file_des, module_name, module)[0m
[1;32m    106[0m   [0;31m# Build messages.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    107[0m   [0;32mfor[0m [0;34m([0m[0mname[0m[0;34m,[0m [0mmsg_des[0m[0;34m)[0m [0;32min[0m [0mfile_des[0m[0;34m.[0m[0mmessage_types_by_name[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 108[0;31m     [0mmodule[0m[0;34m[[0m[0mname[0m[0;34m][0m [0;34m=[0m [0mBuildMessage[0m[0;34m([0m[0mmsg_des[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    109[0m [0;34m[0m[0m
[1;32m    110[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py[0m in [0;36mBuildMessage[0;34m(msg_des)[0m
[1;32m     80[0m     [0mcreate_dict[0m [0;34m=[0m [0;34m{[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[1;32m     81[0m     [0;32mfor[0m [0;34m([0m[0mname[0m[0;34m,[0m [0mnested_msg[0m[0;34m)[0m [0;32min[0m [0mmsg_des[0m[0;34m.[0m[0mnested_types_by_name[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 82[0;31m       [0mcreate_dict[0m[0;34m[[0m[0mname[0m[0;34m][0m [0;34m=[0m [0mBuildMessage[0m[0;34m([0m[0mnested_msg[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     83[0m     [0mcreate_dict[0m[0;34m[[0m[0;34m'DESCRIPTOR'[0m[0;34m][0m [0;34m=[0m [0mmsg_des[0m[0;34m[0m[0;34m[0m[0m
[1;32m     84[0m     [0mcreate_dict[0m[0;34m[[0m[0;34m'__module__'[0m[0;34m][0m [0;34m=[0m [0mmodule_name[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py[0m in [0;36mBuildMessage[0;34m(msg_des)[0m
[1;32m     85[0m     message_class = _reflection.GeneratedProtocolMessageType(
[1;32m     86[0m         msg_des.name, (_message.Message,), create_dict)
[0;32m---> 87[0;31m     [0m_sym_db[0m[0;34m.[0m[0mRegisterMessage[0m[0;34m([0m[0mmessage_class[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     88[0m     [0;32mreturn[0m [0mmessage_class[0m[0;34m[0m[0;34m[0m[0m
[1;32m     89[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/google/protobuf/symbol_database.py[0m in [0;36mRegisterMessage[0;34m(self, message)[0m
[1;32m     81[0m [0;34m[0m[0m
[1;32m     82[0m     [0mdesc[0m [0;34m=[0m [0mmessage[0m[0;34m.[0m[0mDESCRIPTOR[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 83[0;31m     [0mself[0m[0;34m.[0m[0m_classes[0m[0;34m[[0m[0mdesc[0m[0;34m][0m [0;34m=[0m [0mmessage[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     84[0m     [0mself[0m[0;34m.[0m[0mRegisterMessageDescriptor[0m[0;34m([0m[0mdesc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     85[0m     [0;32mreturn[0m [0mmessage[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'SymbolDatabase' object has no attribute '_classes'

## === cell 1
df = pd.read_csv('./../input/train.csv')
a2c = {'EAP': 0, 'HPL' : 1, 'MWS' : 2}
y = np.array([a2c[a] for a in df.author])
y = to_categorical(y)
y
