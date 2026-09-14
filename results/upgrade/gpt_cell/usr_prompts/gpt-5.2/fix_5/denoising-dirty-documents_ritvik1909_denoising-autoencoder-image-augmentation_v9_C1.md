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

3.9

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
seaborn==0.12.2
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
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 4. Code solution

## === cell 0
import os

import sys, subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imgaug==0.4.0"])

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_style("darkgrid")
except Exception:
    sns = None
    plt.style.use("dark_background")

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

import imgaug as ia
from imgaug import augmenters as iaa


## --- ERROR in cell 0, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2636709202.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     27[0m [0;32mimport[0m [0mzipfile[0m[0;34m,[0m [0mcv2[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m [0;32mfrom[0m [0mtqdm[0m[0;34m.[0m[0mauto[0m [0;32mimport[0m [0mtqdm[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 29[0;31m [0;32mimport[0m [0mtensorflow[0m [0;32mas[0m [0mtf[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     30[0m [0;34m[0m[0m
[1;32m     31[0m [0;32mfrom[0m [0mtensorflow[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0mmodels[0m [0;32mimport[0m [0mModel[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m    465[0m     [0mimportlib[0m[0;34m.[0m[0mimport_module[0m[0;34m([0m[0;34m"tf_keras.src.optimizers"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    466[0m   [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 467[0;31m     [0mimportlib[0m[0;34m.[0m[0mimport_module[0m[0;34m([0m[0;34m"keras.src.optimizers"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    468[0m [0;32mexcept[0m [0;34m([0m[0mImportError[0m[0;34m,[0m [0mAttributeError[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    469[0m   [0;32mpass[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    124[0m                 [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m
[1;32m    128[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;31m# DO NOT EDIT. Generated by api_gen.sh[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mDTypePolicy[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mFloatDTypePolicy[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mFunction[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mInitializer[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/api/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     33[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mutils[0m[0;34m[0m[0;34m[0m[0m
[1;32m     34[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mvisualization[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 35[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0mapi[0m [0;32mimport[0m [0mwrappers[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     36[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m [0;32mimport[0m [0mVariable[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mbackend[0m [0;32mimport[0m [0mdevice[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/api/wrappers/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      5[0m """
[1;32m      6[0m [0;34m[0m[0m
[0;32m----> 7[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mwrappers[0m[0;34m.[0m[0msklearn_wrapper[0m [0;32mimport[0m [0mSKLearnClassifier[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mwrappers[0m[0;34m.[0m[0msklearn_wrapper[0m [0;32mimport[0m [0mSKLearnRegressor[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mwrappers[0m[0;34m.[0m[0msklearn_wrapper[0m [0;32mimport[0m [0mSKLearnTransformer[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/wrappers/__init__.py[0m in [0;36m<module>[0;34m[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mwrappers[0m[0;34m.[0m[0msklearn_wrapper[0m [0;32mimport[0m [0mSKLearnClassifier[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mwrappers[0m[0;34m.[0m[0msklearn_wrapper[0m [0;32mimport[0m [0mSKLearnRegressor[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mwrappers[0m[0;34m.[0m[0msklearn_wrapper[0m [0;32mimport[0m [0mSKLearnTransformer[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0m__all__[0m [0;34m=[0m [0;34m[[0m[0;34m"SKLearnClassifier"[0m[0;34m,[0m [0;34m"SKLearnRegressor"[0m[0;34m,[0m [0;34m"SKLearnTransformer"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/wrappers/sklearn_wrapper.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mmodels[0m[0;34m.[0m[0mcloning[0m [0;32mimport[0m [0mclone_model[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mmodels[0m[0;34m.[0m[0mmodel[0m [0;32mimport[0m [0mModel[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mwrappers[0m[0;34m.[0m[0mfixes[0m [0;32mimport[0m [0m_routing_enabled[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mwrappers[0m[0;34m.[0m[0mfixes[0m [0;32mimport[0m [0m_validate_data[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m [0;32mfrom[0m [0mkeras[0m[0;34m.[0m[0msrc[0m[0;34m.[0m[0mwrappers[0m[0;34m.[0m[0mfixes[0m [0;32mimport[0m [0mtype_of_target[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/wrappers/fixes.py[0m in [0;36m<module>[0;34m[0m
[1;32m      1[0m [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m     [0;32mimport[0m [0msklearn[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;32mexcept[0m [0mImportError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0msklearn[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     80[0m     [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m_distributor_init[0m  [0;31m# noqa: F401[0m[0;34m[0m[0;34m[0m[0m
[1;32m     81[0m     [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m__check_build[0m  [0;31m# noqa: F401[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 82[0;31m     [0;32mfrom[0m [0;34m.[0m[0mbase[0m [0;32mimport[0m [0mclone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     83[0m     [0;32mfrom[0m [0;34m.[0m[0mutils[0m[0;34m.[0m[0m_show_versions[0m [0;32mimport[0m [0mshow_versions[0m[0;34m[0m[0;34m[0m[0m
[1;32m     84[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m<module>[0;34m[0m
[1;32m     15[0m [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m__version__[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;32mfrom[0m [0;34m.[0m[0m_config[0m [0;32mimport[0m [0mget_config[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 17[0;31m [0;32mfrom[0m [0;34m.[0m[0mutils[0m [0;32mimport[0m [0m_IS_32BIT[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;32mfrom[0m [0;34m.[0m[0mutils[0m[0;34m.[0m[0m_set_output[0m [0;32mimport[0m [0m_SetOutputMixin[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m from .utils._tags import (

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     23[0m [0;32mfrom[0m [0;34m.[0m[0mdeprecation[0m [0;32mimport[0m [0mdeprecated[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;32mfrom[0m [0;34m.[0m[0mdiscovery[0m [0;32mimport[0m [0mall_estimators[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 25[0;31m [0;32mfrom[0m [0;34m.[0m[0mfixes[0m [0;32mimport[0m [0mparse_version[0m[0;34m,[0m [0mthreadpool_info[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     26[0m [0;32mfrom[0m [0;34m.[0m[0m_estimator_html_repr[0m [0;32mimport[0m [0mestimator_html_repr[0m[0;34m[0m[0;34m[0m[0m
[1;32m     27[0m from .validation import (

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/fixes.py[0m in [0;36m<module>[0;34m[0m
[1;32m     17[0m [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;32mimport[0m [0mscipy[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 19[0;31m [0;32mimport[0m [0mscipy[0m[0;34m.[0m[0mstats[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;32mimport[0m [0mthreadpoolctl[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/stats/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m    622[0m from ._warnings_errors import (ConstantInputWarning, NearConstantInputWarning,
[1;32m    623[0m                                DegenerateDataWarning, FitError)
[0;32m--> 624[0;31m [0;32mfrom[0m [0;34m.[0m[0m_stats_py[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    625[0m [0;32mfrom[0m [0;34m.[0m[0m_variation[0m [0;32mimport[0m [0mvariation[0m[0;34m[0m[0;34m[0m[0m
[1;32m    626[0m [0;32mfrom[0m [0;34m.[0m[0mdistributions[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/stats/_stats_py.py[0m in [0;36m<module>[0;34m[0m
[1;32m     37[0m [0;34m[0m[0m
[1;32m     38[0m [0;32mfrom[0m [0mscipy[0m [0;32mimport[0m [0msparse[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 39[0;31m [0;32mfrom[0m [0mscipy[0m[0;34m.[0m[0mspatial[0m [0;32mimport[0m [0mdistance_matrix[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     40[0m [0;34m[0m[0m
[1;32m     41[0m [0;32mfrom[0m [0mscipy[0m[0;34m.[0m[0moptimize[0m [0;32mimport[0m [0mmilp[0m[0;34m,[0m [0mLinearConstraint[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/spatial/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m    114[0m [0;32mfrom[0m [0;34m.[0m[0m_plotutils[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m
[1;32m    115[0m [0;32mfrom[0m [0;34m.[0m[0m_procrustes[0m [0;32mimport[0m [0mprocrustes[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 116[0;31m [0;32mfrom[0m [0;34m.[0m[0m_geometric_slerp[0m [0;32mimport[0m [0mgeometric_slerp[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    117[0m [0;34m[0m[0m
[1;32m    118[0m [0;31m# Deprecated namespaces, to be removed in v2.0.0[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/spatial/_geometric_slerp.py[0m in [0;36m<module>[0;34m[0m
[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0;32mimport[0m [0mnumpy[0m [0;32mas[0m [0mnp[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m [0;32mfrom[0m [0mscipy[0m[0;34m.[0m[0mspatial[0m[0;34m.[0m[0mdistance[0m [0;32mimport[0m [0meuclidean[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0;32mif[0m [0mTYPE_CHECKING[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/spatial/distance.py[0m in [0;36m<module>[0;34m[0m
[1;32m    119[0m [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m_hausdorff[0m[0;34m[0m[0;34m[0m[0m
[1;32m    120[0m [0;32mfrom[0m [0;34m.[0m[0;34m.[0m[0mlinalg[0m [0;32mimport[0m [0mnorm[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 121[0;31m [0;32mfrom[0m [0;34m.[0m[0;34m.[0m[0mspecial[0m [0;32mimport[0m [0mrel_entr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    122[0m [0;34m[0m[0m
[1;32m    123[0m [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m_distance_pybind[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/special/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m    824[0m     chdtr, chdtrc, betainc, betaincc, stdtr)
[1;32m    825[0m [0;34m[0m[0m
[0;32m--> 826[0;31m [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m_basic[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    827[0m [0;32mfrom[0m [0;34m.[0m[0m_basic[0m [0;32mimport[0m [0;34m*[0m[0;34m[0m[0;34m[0m[0m
[1;32m    828[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/special/_basic.py[0m in [0;36m<module>[0;34m[0m
[1;32m     20[0m [0;32mfrom[0m [0;34m.[0m [0;32mimport[0m [0m_specfun[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m [0;32mfrom[0m [0;34m.[0m[0m_comb[0m [0;32mimport[0m [0m_comb_int[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 22[0;31m from ._multiufuncs import (assoc_legendre_p_all,
[0m[1;32m     23[0m                            legendre_p_all)
[1;32m     24[0m [0;32mfrom[0m [0mscipy[0m[0;34m.[0m[0m_lib[0m[0;34m.[0m[0mdeprecation[0m [0;32mimport[0m [0m_deprecated[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py[0m in [0;36m<module>[0;34m[0m
[1;32m    140[0m [0;34m[0m[0m
[1;32m    141[0m [0;34m[0m[0m
[0;32m--> 142[0;31m sph_legendre_p = MultiUFunc(
[0m[1;32m    143[0m     [0msph_legendre_p[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    144[0m     r"""sph_legendre_p(n, m, theta, *, diff_n=0)

[0;32m/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py[0m in [0;36m__init__[0;34m(self, ufunc_or_ufuncs, doc, force_complex_output, **default_kwargs)[0m
[1;32m     39[0m             [0;32mfor[0m [0mufunc[0m [0;32min[0m [0mufuncs_iter[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m                 [0;32mif[0m [0;32mnot[0m [0misinstance[0m[0;34m([0m[0mufunc[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mufunc[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 41[0;31m                     raise ValueError("All ufuncs must have type `numpy.ufunc`."
[0m[1;32m     42[0m                                      f" Received {ufunc_or_ufuncs}")
[1;32m     43[0m                 [0mseen_input_types[0m[0;34m.[0m[0madd[0m[0;34m([0m[0mfrozenset[0m[0;34m([0m[0mx[0m[0;34m.[0m[0msplit[0m[0;34m([0m[0;34m"->"[0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m [0;32mfor[0m [0mx[0m [0;32min[0m [0mufunc[0m[0;34m.[0m[0mtypes[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)

## === cell 1
path_zip = '../input/denoising-dirty-documents/'
path = '/kaggle/working/'

with zipfile.ZipFile(path_zip + 'train.zip', 'r') as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + 'test.zip', 'r') as zip_ref:
    zip_ref.extractall(path)  
    
with zipfile.ZipFile(path_zip + 'train_cleaned.zip', 'r') as zip_ref:
    zip_ref.extractall(path)  
    
with zipfile.ZipFile(path_zip + 'sampleSubmission.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(path)
    
train_img = sorted(os.listdir(path + '/train'))
train_cleaned_img = sorted(os.listdir(path + '/train_cleaned'))
test_img = sorted(os.listdir(path + '/test'))
