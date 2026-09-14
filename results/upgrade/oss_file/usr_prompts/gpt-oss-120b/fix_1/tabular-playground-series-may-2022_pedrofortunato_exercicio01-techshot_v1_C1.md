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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
ydata-profiling==4.17.0
yellowbrick==1.5

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.81171

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install distfit
!pip install fastai
!pip install autoviz
!pip install pandas-profiling


## === cell 1
!pip install evidently
!pip install fairlearn
!pip install lime


## === cell 2
import missingno as msno
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from distfit import distfit
from fastai.tabular.core import cont_cat_split
import scipy
import seaborn as sns
from autoviz.AutoViz_Class import AutoViz_Class
from pandas_profiling import ProfileReport
from yellowbrick.target import class_balance
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve as pr_curve

from sklearn.metrics import (roc_auc_score, 
                             accuracy_score,
                             precision_score,
                             recall_score,
                             roc_curve)

from yellowbrick.classifier import confusion_matrix
from yellowbrick.classifier import classification_report
from yellowbrick.classifier.rocauc import roc_auc
from yellowbrick.classifier import precision_recall_curve
from yellowbrick.classifier import class_prediction_error
from yellowbrick.model_selection import validation_curve
from yellowbrick.model_selection import feature_importances
from yellowbrick.contrib.classifier import DecisionViz



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1205503820.py in <cell line: 0>()
----> 1 import missingno as msno
      2 import pandas as pd
      3 import numpy as np
      4 import plotly.express as px
      5 import plotly.graph_objects as go

/usr/local/lib/python3.11/dist-packages/missingno/__init__.py in <module>
----> 1 from .missingno import matrix
      2 from .missingno import bar
      3 from .missingno import heatmap
      4 from .missingno import dendrogram
      5 from .missingno import nullity_filter

/usr/local/lib/python3.11/dist-packages/missingno/missingno.py in <module>
      3 from matplotlib import gridspec
      4 import matplotlib.pyplot as plt
----> 5 from scipy.cluster import hierarchy
      6 import seaborn as sns
      7 import pandas as pd

/usr/local/lib/python3.11/dist-packages/scipy/cluster/__init__.py in <module>
     25 __all__ = ['vq', 'hierarchy']
     26 
---> 27 from . import vq, hierarchy
     28 
     29 from scipy._lib._testutils import PytestTester

/usr/local/lib/python3.11/dist-packages/scipy/cluster/vq.py in <module>
     74                               _transition_to_rng)
     75 from scipy._lib import array_api_extra as xpx
---> 76 from scipy.spatial.distance import cdist
     77 
     78 from . import _vq

/usr/local/lib/python3.11/dist-packages/scipy/spatial/__init__.py in <module>
    114 from ._plotutils import *
    115 from ._procrustes import procrustes
--> 116 from ._geometric_slerp import geometric_slerp
    117 
    118 # Deprecated namespaces, to be removed in v2.0.0

/usr/local/lib/python3.11/dist-packages/scipy/spatial/_geometric_slerp.py in <module>
      5 
      6 import numpy as np
----> 7 from scipy.spatial.distance import euclidean
      8 
      9 if TYPE_CHECKING:

/usr/local/lib/python3.11/dist-packages/scipy/spatial/distance.py in <module>
    119 from . import _hausdorff
    120 from ..linalg import norm
--> 121 from ..special import rel_entr
    122 
    123 from . import _distance_pybind

/usr/local/lib/python3.11/dist-packages/scipy/special/__init__.py in <module>
    824     chdtr, chdtrc, betainc, betaincc, stdtr)
    825 
--> 826 from . import _basic
    827 from ._basic import *
    828 

/usr/local/lib/python3.11/dist-packages/scipy/special/_basic.py in <module>
     20 from . import _specfun
     21 from ._comb import _comb_int
---> 22 from ._multiufuncs import (assoc_legendre_p_all,
     23                            legendre_p_all)
     24 from scipy._lib.deprecation import _deprecated

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in <module>
    140 
    141 
--> 142 sph_legendre_p = MultiUFunc(
    143     sph_legendre_p,
    144     r"""sph_legendre_p(n, m, theta, *, diff_n=0)

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in __init__(self, ufunc_or_ufuncs, doc, force_complex_output, **default_kwargs)
     39             for ufunc in ufuncs_iter:
     40                 if not isinstance(ufunc, np.ufunc):
---> 41                     raise ValueError("All ufuncs must have type `numpy.ufunc`."
     42                                      f" Received {ufunc_or_ufuncs}")
     43                 seen_input_types.add(frozenset(x.split("->")[0] for x in ufunc.types))

ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)

## === cell 3
df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
df_test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
print(df.shape)
df.head(10)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2096796650.py in <cell line: 0>()
----> 1 df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
      2 df_test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
      3 print(df.shape)
      4 df.head(10)

NameError: name 'pd' is not defined

## === cell 4
df.info()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3089318408.py in <cell line: 0>()
----> 1 df.info()

NameError: name 'df' is not defined

## === cell 5

def basic_info(df: pd.DataFrame) -> str:
    s = ""
    count = 1

    for c in df.columns:
        nan = df[c].isna().sum()
        shape = df.shape[0]

        s += f'{count} - {c}\n'
        s += f'   Type: {df[c].dtype}\n'
        s += f'   No. of unique values {df[c].nunique()}\n'
        s += f"    Sample of unique values {df[c].unique()[0:10]}\n"
        s += f"    Null values {nan} ({np.round((nan/shape)*100, 2)} %)\n"
        s += "\n"
        count += 1  
    return s

print(basic_info(df))


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2809432222.py in <cell line: 0>()
      1 #  Função responsável por aprensentar as informações básicas sobre o dataset.
      2 
----> 3 def basic_info(df: pd.DataFrame) -> str:
      4     s = ""
      5     count = 1

NameError: name 'pd' is not defined

## === cell 6
df.drop('id', axis = 1, inplace = True)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3415934169.py in <cell line: 0>()
----> 1 df.drop('id', axis = 1, inplace = True)

NameError: name 'df' is not defined

## === cell 7
cont_names, cat_names = cont_cat_split(df)
print(f'Continuous columns: {cont_names} \n\n Categorical columns: {cat_names}')


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3901674395.py in <cell line: 0>()
----> 1 cont_names, cat_names = cont_cat_split(df)
      2 print(f'Continuous columns: {cont_names} \n\n Categorical columns: {cat_names}')

NameError: name 'cont_cat_split' is not defined

## === cell 8
def skewness_function(df, c):
  

    skewness = scipy.stats.skew(df[c], axis = 0)
    skewness_title = 'Distribuição perto do normal'

    if skewness < -1e-1:
        skewness_title = 'Distorção mais a direita'
    elif skewness > 1e1:
        skewness_title = 'Distorição mais a esquerda'
    return skewness, skewness_title

def kurtosis_function(df, c):
    
    kts = scipy.stats.kurtosis(df[c], axis = 0)
    kts_title = 'Meoskurtic'

    if kts > 3.1:
        kts_name = 'Leptokurtic'
    elif kts < 2.9:
        kts_name = 'Platykurtic'
    return kts, kts_name


## === cell 9
for col in cont_names:
    print(f'COLUNA {col}\n\n')
    
    plt.figure(figsize=(10, 5))
    sns.displot(data = df, x = col, hue = 'target')
    plt.legend(loc='upper right')
    plt.show()

    skewness, skewness_title = skewness_function(df, col)
    print(f'Skewness: {skewness} -> {skewness_title}.\n\n')

    kts, kts_title = kurtosis_function(df, col)
    print(f'Kurtosis: {kts} -> {kts_title}.\n\n')


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3984861656.py in <cell line: 0>()
----> 1 for col in cont_names:
      2     print(f'COLUNA {col}\n\n')
      3 
      4     plt.figure(figsize=(10, 5))
      5     sns.displot(data = df, x = col, hue = 'target')

NameError: name 'cont_names' is not defined

## === cell 10
correlation_num = abs(df[cont_names].corr())
correlation_num


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2431645371.py in <cell line: 0>()
----> 1 correlation_num = abs(df[cont_names].corr())
      2 correlation_num

NameError: name 'df' is not defined

## === cell 11
fig = px.imshow(correlation_num)
fig.show()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1858740469.py in <cell line: 0>()
----> 1 fig = px.imshow(correlation_num)
      2 fig.show()

NameError: name 'px' is not defined

## === cell 12
for c in correlation_num.columns:
    correlation_num.loc[correlation_num[c] < 0.2, c] = 0
    correlation_num.loc[(correlation_num[c] >= 0.2) & (correlation_num[c] < 0.4), c] = 0.25
    correlation_num.loc[(correlation_num[c] >= 0.4) & (correlation_num[c] < 0.6), c] = 0.5
    correlation_num.loc[(correlation_num[c] >= 0.6) & (correlation_num[c] < 0.8), c] = 0.75
    correlation_num.loc[correlation_num[c] >= 0.8, c] = 1
correlation_num


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3332128017.py in <cell line: 0>()
----> 1 for c in correlation_num.columns:
      2     correlation_num.loc[correlation_num[c] < 0.2, c] = 0
      3     correlation_num.loc[(correlation_num[c] >= 0.2) & (correlation_num[c] < 0.4), c] = 0.25
      4     correlation_num.loc[(correlation_num[c] >= 0.4) & (correlation_num[c] < 0.6), c] = 0.5
      5     correlation_num.loc[(correlation_num[c] >= 0.6) & (correlation_num[c] < 0.8), c] = 0.75

NameError: name 'correlation_num' is not defined

## === cell 13
fig = px.imshow(correlation_num)
fig.show()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1858740469.py in <cell line: 0>()
----> 1 fig = px.imshow(correlation_num)
      2 fig.show()

NameError: name 'px' is not defined

## === cell 14
df_copy = df.copy()
df_copy['col'] = 1

for col in cat_names:
    if col != 'target':
        print(f'\n\nCOLUNA: {col}\n')
        ag_df = df_copy.groupby([col]).agg('count')[['col']].sort_values(['col'], ascending = False)
        ag_df = ag_df.reset_index()
        ag_df['cumsum'] = ag_df['col'].cumsum()
        ag_df['percent'] = ag_df['cumsum'] / (ag_df['col'].sum())

        display(ag_df)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4130280130.py in <cell line: 0>()
----> 1 df_copy = df.copy()
      2 df_copy['col'] = 1
      3 
      4 for col in cat_names:
      5     if col != 'target':

NameError: name 'df' is not defined

## === cell 15
from autoviz.AutoViz_Utils import train_test_split
X = df.drop(columns=['target','f_27'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
print(f'X_train: {X_train.shape}. \nX_test: {X_test.shape}. \n\ny_train: {y_train.shape}. \ny_test: {y_test.shape}.')


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/977175042.py in <cell line: 0>()
----> 1 from autoviz.AutoViz_Utils import train_test_split
      2 X = df.drop(columns=['target','f_27'])
      3 y = df['target']
      4 
      5 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

/usr/local/lib/python3.11/dist-packages/autoviz/__init__.py in <module>
      1 name = "autoviz"
      2 from .__version__ import __version__, __holo_version__
----> 3 from .AutoViz_Class import AutoViz_Class
      4 from .AutoViz_Class import data_cleaning_suggestions
      5 from .AutoViz_Class import FixDQ

/usr/local/lib/python3.11/dist-packages/autoviz/AutoViz_Class.py in <module>
     18 ########################################
     19 import warnings
---> 20 from sklearn.exceptions import DataConversionWarning
     21 ####################################################################################
     22 import matplotlib

/usr/local/lib/python3.11/dist-packages/sklearn/__init__.py in <module>
     80     from . import _distributor_init  # noqa: F401
     81     from . import __check_build  # noqa: F401
---> 82     from .base import clone
     83     from .utils._show_versions import show_versions
     84 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in <module>
     15 from . import __version__
     16 from ._config import get_config
---> 17 from .utils import _IS_32BIT
     18 from .utils._set_output import _SetOutputMixin
     19 from .utils._tags import (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/__init__.py in <module>
     23 from .deprecation import deprecated
     24 from .discovery import all_estimators
---> 25 from .fixes import parse_version, threadpool_info
     26 from ._estimator_html_repr import estimator_html_repr
     27 from .validation import (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/fixes.py in <module>
     17 import numpy as np
     18 import scipy
---> 19 import scipy.stats
     20 import threadpoolctl
     21 

/usr/local/lib/python3.11/dist-packages/scipy/stats/__init__.py in <module>
    622 from ._warnings_errors import (ConstantInputWarning, NearConstantInputWarning,
    623                                DegenerateDataWarning, FitError)
--> 624 from ._stats_py import *
    625 from ._variation import variation
    626 from .distributions import *

/usr/local/lib/python3.11/dist-packages/scipy/stats/_stats_py.py in <module>
     37 
     38 from scipy import sparse
---> 39 from scipy.spatial import distance_matrix
     40 
     41 from scipy.optimize import milp, LinearConstraint

/usr/local/lib/python3.11/dist-packages/scipy/spatial/__init__.py in <module>
    114 from ._plotutils import *
    115 from ._procrustes import procrustes
--> 116 from ._geometric_slerp import geometric_slerp
    117 
    118 # Deprecated namespaces, to be removed in v2.0.0

/usr/local/lib/python3.11/dist-packages/scipy/spatial/_geometric_slerp.py in <module>
      5 
      6 import numpy as np
----> 7 from scipy.spatial.distance import euclidean
      8 
      9 if TYPE_CHECKING:

/usr/local/lib/python3.11/dist-packages/scipy/spatial/distance.py in <module>
    119 from . import _hausdorff
    120 from ..linalg import norm
--> 121 from ..special import rel_entr
    122 
    123 from . import _distance_pybind

/usr/local/lib/python3.11/dist-packages/scipy/special/__init__.py in <module>
    824     chdtr, chdtrc, betainc, betaincc, stdtr)
    825 
--> 826 from . import _basic
    827 from ._basic import *
    828 

/usr/local/lib/python3.11/dist-packages/scipy/special/_basic.py in <module>
     20 from . import _specfun
     21 from ._comb import _comb_int
---> 22 from ._multiufuncs import (assoc_legendre_p_all,
     23                            legendre_p_all)
     24 from scipy._lib.deprecation import _deprecated

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in <module>
    140 
    141 
--> 142 sph_legendre_p = MultiUFunc(
    143     sph_legendre_p,
    144     r"""sph_legendre_p(n, m, theta, *, diff_n=0)

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in __init__(self, ufunc_or_ufuncs, doc, force_complex_output, **default_kwargs)
     39             for ufunc in ufuncs_iter:
     40                 if not isinstance(ufunc, np.ufunc):
---> 41                     raise ValueError("All ufuncs must have type `numpy.ufunc`."
     42                                      f" Received {ufunc_or_ufuncs}")
     43                 seen_input_types.add(frozenset(x.split("->")[0] for x in ufunc.types))

ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)

## === cell 16
from sklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier(random_state = 42, max_depth = 14)
d_tree.fit(X_train, y_train)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1177857532.py in <cell line: 0>()
----> 1 from sklearn.tree import DecisionTreeClassifier
      2 d_tree = DecisionTreeClassifier(random_state = 42, max_depth = 14)
      3 d_tree.fit(X_train, y_train)

/usr/local/lib/python3.11/dist-packages/sklearn/__init__.py in <module>
     80     from . import _distributor_init  # noqa: F401
     81     from . import __check_build  # noqa: F401
---> 82     from .base import clone
     83     from .utils._show_versions import show_versions
     84 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in <module>
     15 from . import __version__
     16 from ._config import get_config
---> 17 from .utils import _IS_32BIT
     18 from .utils._set_output import _SetOutputMixin
     19 from .utils._tags import (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/__init__.py in <module>
     23 from .deprecation import deprecated
     24 from .discovery import all_estimators
---> 25 from .fixes import parse_version, threadpool_info
     26 from ._estimator_html_repr import estimator_html_repr
     27 from .validation import (

/usr/local/lib/python3.11/dist-packages/sklearn/utils/fixes.py in <module>
     17 import numpy as np
     18 import scipy
---> 19 import scipy.stats
     20 import threadpoolctl
     21 

/usr/local/lib/python3.11/dist-packages/scipy/stats/__init__.py in <module>
    622 from ._warnings_errors import (ConstantInputWarning, NearConstantInputWarning,
    623                                DegenerateDataWarning, FitError)
--> 624 from ._stats_py import *
    625 from ._variation import variation
    626 from .distributions import *

/usr/local/lib/python3.11/dist-packages/scipy/stats/_stats_py.py in <module>
     37 
     38 from scipy import sparse
---> 39 from scipy.spatial import distance_matrix
     40 
     41 from scipy.optimize import milp, LinearConstraint

/usr/local/lib/python3.11/dist-packages/scipy/spatial/__init__.py in <module>
    114 from ._plotutils import *
    115 from ._procrustes import procrustes
--> 116 from ._geometric_slerp import geometric_slerp
    117 
    118 # Deprecated namespaces, to be removed in v2.0.0

/usr/local/lib/python3.11/dist-packages/scipy/spatial/_geometric_slerp.py in <module>
      5 
      6 import numpy as np
----> 7 from scipy.spatial.distance import euclidean
      8 
      9 if TYPE_CHECKING:

/usr/local/lib/python3.11/dist-packages/scipy/spatial/distance.py in <module>
    119 from . import _hausdorff
    120 from ..linalg import norm
--> 121 from ..special import rel_entr
    122 
    123 from . import _distance_pybind

/usr/local/lib/python3.11/dist-packages/scipy/special/__init__.py in <module>
    824     chdtr, chdtrc, betainc, betaincc, stdtr)
    825 
--> 826 from . import _basic
    827 from ._basic import *
    828 

/usr/local/lib/python3.11/dist-packages/scipy/special/_basic.py in <module>
     20 from . import _specfun
     21 from ._comb import _comb_int
---> 22 from ._multiufuncs import (assoc_legendre_p_all,
     23                            legendre_p_all)
     24 from scipy._lib.deprecation import _deprecated

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in <module>
    140 
    141 
--> 142 sph_legendre_p = MultiUFunc(
    143     sph_legendre_p,
    144     r"""sph_legendre_p(n, m, theta, *, diff_n=0)

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in __init__(self, ufunc_or_ufuncs, doc, force_complex_output, **default_kwargs)
     39             for ufunc in ufuncs_iter:
     40                 if not isinstance(ufunc, np.ufunc):
---> 41                     raise ValueError("All ufuncs must have type `numpy.ufunc`."
     42                                      f" Received {ufunc_or_ufuncs}")
     43                 seen_input_types.add(frozenset(x.split("->")[0] for x in ufunc.types))

ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)

## === cell 17
y_pred = d_tree.predict(X_test)
y_pred


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2833725959.py in <cell line: 0>()
----> 1 y_pred = d_tree.predict(X_test)
      2 y_pred

NameError: name 'd_tree' is not defined

## === cell 18
y_proba = d_tree.predict_proba(X_test)
y_proba


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3370665749.py in <cell line: 0>()
----> 1 y_proba = d_tree.predict_proba(X_test)
      2 y_proba

NameError: name 'd_tree' is not defined

## === cell 19
accuracy_score(y_test, y_pred)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/582378252.py in <cell line: 0>()
----> 1 accuracy_score(y_test, y_pred)

NameError: name 'accuracy_score' is not defined

## === cell 20
dummy_clf = DummyClassifier(constant=1, strategy="constant")

dummy_clf.fit(X_train, y_train)
y_pred_dummy = dummy_clf.predict(X_test)
accuracy_score(y_test, y_pred_dummy)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/559684627.py in <cell line: 0>()
----> 1 dummy_clf = DummyClassifier(constant=1, strategy="constant")
      2 
      3 dummy_clf.fit(X_train, y_train)
      4 y_pred_dummy = dummy_clf.predict(X_test)
      5 accuracy_score(y_test, y_pred_dummy)

NameError: name 'DummyClassifier' is not defined

## === cell 21
dummy_clf = DummyClassifier(strategy="uniform")

dummy_clf.fit(X_train, y_train)
y_pred_dummy = dummy_clf.predict(X_test)
accuracy_score(y_test, y_pred_dummy)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1812695676.py in <cell line: 0>()
----> 1 dummy_clf = DummyClassifier(strategy="uniform")
      2 
      3 dummy_clf.fit(X_train, y_train)
      4 y_pred_dummy = dummy_clf.predict(X_test)
      5 accuracy_score(y_test, y_pred_dummy)

NameError: name 'DummyClassifier' is not defined

## === cell 22
dummy_clf = DummyClassifier(strategy="stratified")

dummy_clf.fit(X_train, y_train)
y_pred_dummy = dummy_clf.predict(X_test)
accuracy_score(y_test, y_pred_dummy)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1811127779.py in <cell line: 0>()
----> 1 dummy_clf = DummyClassifier(strategy="stratified")
      2 
      3 dummy_clf.fit(X_train, y_train)
      4 y_pred_dummy = dummy_clf.predict(X_test)
      5 accuracy_score(y_test, y_pred_dummy)

NameError: name 'DummyClassifier' is not defined

## === cell 23
dummy_clf = DummyClassifier(constant=0, strategy="constant")

dummy_clf.fit(X_train, y_train)
y_pred_dummy = dummy_clf.predict(X_test)
accuracy_score(y_test, y_pred_dummy)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3529771536.py in <cell line: 0>()
----> 1 dummy_clf = DummyClassifier(constant=0, strategy="constant")
      2 
      3 dummy_clf.fit(X_train, y_train)
      4 y_pred_dummy = dummy_clf.predict(X_test)
      5 accuracy_score(y_test, y_pred_dummy)

NameError: name 'DummyClassifier' is not defined

## === cell 24
plt.figure(figsize=(6, 6))
confusion_matrix(
    d_tree,
    X_train, y_train, X_test, y_test,
    percent=False
)
plt.tight_layout()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/109407364.py in <cell line: 0>()
----> 1 plt.figure(figsize=(6, 6))
      2 confusion_matrix(
      3     d_tree,
      4     X_train, y_train, X_test, y_test,
      5     percent=False

NameError: name 'plt' is not defined

## === cell 25
plt.figure(figsize=(6, 6))
confusion_matrix(
    d_tree,
    X_train, y_train, X_test, y_test,
    percent=True
)
plt.tight_layout()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1893488679.py in <cell line: 0>()
----> 1 plt.figure(figsize=(6, 6))
      2 confusion_matrix(
      3     d_tree,
      4     X_train, y_train, X_test, y_test,
      5     percent=True

NameError: name 'plt' is not defined

## === cell 26
plt.figure(figsize=(6, 6))
class_prediction_error(
    d_tree,
    X_train, y_train, X_test, y_test
)
plt.tight_layout()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1062787313.py in <cell line: 0>()
----> 1 plt.figure(figsize=(6, 6))
      2 class_prediction_error(
      3     d_tree,
      4     X_train, y_train, X_test, y_test
      5 )

NameError: name 'plt' is not defined

## === cell 27
ground_truth = pd.DataFrame(y_test.values)
res_0 = []
res_1 = []

for i, p in enumerate(y_proba[:, 1]):
    if y_test.iloc[i] > 0:
        res_1.append(p)
    else:
        res_0.append(p)

plt.axvline(0.5, 0, 2.5)
sns.distplot(res_0, hist=False, kde_kws={"shade": True}, color='b', label='0')
sns.distplot(res_1, hist=False, kde_kws={"shade": True}, color='r', label='1')
plt.legend()
plt.show()


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2335469428.py in <cell line: 0>()
----> 1 ground_truth = pd.DataFrame(y_test.values)
      2 res_0 = []
      3 res_1 = []
      4 
      5 for i, p in enumerate(y_proba[:, 1]):

NameError: name 'pd' is not defined

## === cell 28
plt.figure(figsize=(10, 6))
classification_report(
    d_tree, X_train, y_train, X_test, y_test, support=True
)
plt.tight_layout()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4012513839.py in <cell line: 0>()
----> 1 plt.figure(figsize=(10, 6))
      2 classification_report(
      3     d_tree, X_train, y_train, X_test, y_test, support=True
      4 )
      5 plt.tight_layout()

NameError: name 'plt' is not defined

## === cell 29
plt.figure(figsize=(6, 6))
roc_auc(d_tree, X_train, y_train, X_test=X_test, y_test=y_test)
plt.tight_layout()


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/72075887.py in <cell line: 0>()
----> 1 plt.figure(figsize=(6, 6))
      2 roc_auc(d_tree, X_train, y_train, X_test=X_test, y_test=y_test)
      3 plt.tight_layout()

NameError: name 'plt' is not defined

## === cell 30
plt.figure(figsize=(6, 6))
precision_recall_curve(d_tree, X_train, y_train, X_test, y_test)
plt.tight_layout()


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3821429351.py in <cell line: 0>()
----> 1 plt.figure(figsize=(6, 6))
      2 precision_recall_curve(d_tree, X_train, y_train, X_test, y_test)
      3 plt.tight_layout()

NameError: name 'plt' is not defined

## === cell 31
test_df = df_test.drop(columns=['id','f_27'])
my_predictions = d_tree.predict_proba(test_df)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1528422435.py in <cell line: 0>()
----> 1 test_df = df_test.drop(columns=['id','f_27'])
      2 my_predictions = d_tree.predict_proba(test_df)

NameError: name 'df_test' is not defined

## === cell 32
y_proba = d_tree.predict_proba(test_df)
y_proba


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/668191736.py in <cell line: 0>()
----> 1 y_proba = d_tree.predict_proba(test_df)
      2 y_proba

NameError: name 'd_tree' is not defined

## === cell 33
submission = pd.DataFrame({'id': df_test.id, 'target': y_proba[:, 1]})
submission.head()


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4033670844.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({'id': df_test.id, 'target': y_proba[:, 1]})
      2 submission.head()

NameError: name 'pd' is not defined

## === cell 34
submission.to_csv('submission_example.csv', index=False)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3192119877.py in <cell line: 0>()
----> 1 submission.to_csv('submission_example.csv', index=False)

NameError: name 'submission' is not defined
