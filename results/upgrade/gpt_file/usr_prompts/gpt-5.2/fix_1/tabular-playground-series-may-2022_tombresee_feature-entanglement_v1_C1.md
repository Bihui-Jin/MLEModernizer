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

3.10

# 3. Installed packages



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

0.93372

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%%HTML
<style type="text/css">
     
div.h2 { background-color: #159957;
         background-image: linear-gradient(120deg, #155799, #159957);
         text-align: left;
         color: white;              
         padding:9px;
         padding-right: 100px; 
         font-size: 20px; 
         max-width: 1500px; 
         margin: auto; 
         margin-top: 40px; }
                                                                  
body {font-size: 12px;}    
     
                                    
                                      
div.h3 {color: #159957; 
        font-size: 18px; 
        margin-top: 20px; 
        margin-bottom:4px;}
   
                                      
div.h4 {color: #159957;
        font-size: 15px; 
        margin-top: 20px; 
        margin-bottom: 8px;}
                                           
                                      
span.note {font-size: 5;
           color: gray; 
           font-style: italic;}
  
                                      
hr {
    display: block; 
    color: gray
    height: 1px; 
    border: 0; 
    border-top: 1px solid;
}
  
                                      
hr.light {
    display: block; 
    color: lightgray
    height: 1px; 
    border: 0; 
    border-top: 1px solid;
}   
    
                                      
table.dataframe th 
{
    border: 1px darkgray solid;
    color: black;
       align="left">
    ...
  
    background-color: white;
}
    
                                      
table.dataframe td 
{
    border: 1px darkgray solid;
    color: black;
    background-color: white;
    font-size: 11px;
    text-align: center;
} 
          
                                      
table.rules th 
{
    border: 1px darkgray solid;
    color: black;
    background-color: white;
    font-size: 11px;
    align: left;
}
       
                                      
table.rules td 
{
    border: 1px darkgray solid;
    color: black;
    background-color: white;
    font-size: 13px;
    text-align: center;
} 
   
                                      
                                      
table.rules tr.best
{
    color: green;
}    
    
                                      
.output { 
    align-items: left; 
}
        
                                      
.output_png {
    display: table-cell;
    text-align: left;
    margin:auto;
}                                          
                                                                                                     
                                      
</style>  


## === cell 1
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_context('notebook')
from cycler import cycler
from IPython.display import display
import datetime
from io import StringIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib_inline.backend_inline
from IPython.display import set_matplotlib_formats
import matplotlib
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
plt.rcParams['savefig.facecolor']='white'
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
plt.rcParams['figure.dpi'] = 300
from sklearn.metrics import roc_auc_score
from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot  #firstly import gridplot
output_notebook()
from xgboost  import XGBClassifier
train_df = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test_df = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
sub_df = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
columns_int   = train_df.select_dtypes(include=['int']).columns
columns_float = train_df.select_dtypes(include=['float']).columns


## === cell 2
data = ("""ID,Parameter,Value,Description
1, - Total Observations - ,"1,600,000",train and test sets combined
2,Observations (train),"900,000", 56.25% of total observations
3,Observations (test),"700,000",43.75% of total observations
4, - Total Features -,32,(does not include the target)
5,Features (Integers),15, Note: One of the integer features is 'id' - a tagged unique value for every observations (helpful)
6,Features (Float), 16,ex
7, Features (Object/String), 1, ex: ACACCADCEB (always 10 characters)
8, Missing Values, 0, (doesn't appear to be any missing values)
9,Submission Criteria, AUC, Submissions evaluated on area under the ROC curve (between the predicted probability and the observed target)
""")

mapper = pd.read_csv(StringIO(data))

d = dict(selector="th", props=[('text-align', 'center')])

mapper = mapper.set_index('ID')

mapper.style.set_properties(**{'width':'18em', 'text-align':'center'})\
        .set_table_styles([d])


## === cell 3
display(train_df.head(4).T)


## === cell 4
sns.set_style("white")

fig, axs = plt.subplots(4, 4, figsize=(17,17))

sns.histplot(data=train_df, x="f_00", kde=True, color ='xkcd:lightish blue', ax=axs[0, 0])
    
sns.histplot(data=train_df, x="f_01", kde=True, color ='xkcd:lightish blue', ax=axs[0, 1])

sns.histplot(data=train_df, x="f_02", kde=True, color ='xkcd:lightish blue',  ax=axs[0, 2])

sns.histplot(data=train_df, x="f_03", kde=True, color ='xkcd:lightish blue',  ax=axs[0, 3])

sns.histplot(data=train_df, x="f_04", kde=True, color ='xkcd:lightish blue', ax=axs[1, 0])

sns.histplot(data=train_df, x="f_05", kde=True, color ='xkcd:lightish blue', ax=axs[1, 1])

sns.histplot(data=train_df, x="f_06", kde=True, color ='xkcd:lightish blue', ax=axs[1, 2])

sns.histplot(data=train_df, x="f_19", kde=True, color ='xkcd:lightish blue', ax=axs[1, 3])

sns.histplot(data=train_df, x="f_20", kde=True, color ='xkcd:lightish blue', ax=axs[2, 0])

sns.histplot(data=train_df, x="f_21", kde=True, color ='xkcd:lightish blue', ax=axs[2, 1])

sns.histplot(data=train_df, x="f_22", kde=True, color ='xkcd:lightish blue', ax=axs[2, 2])

sns.histplot(data=train_df, x="f_23", kde=True, color ='xkcd:lightish blue', ax=axs[2, 3])

sns.histplot(data=train_df, x="f_24", kde=True, color ='xkcd:lightish blue', ax=axs[3, 0])

sns.histplot(data=train_df, x="f_25", kde=True, color ='xkcd:lightish blue', ax=axs[3, 1])

sns.histplot(data=train_df, x="f_26", kde=True, color ='xkcd:lightish blue', ax=axs[3, 2])

sns.histplot(data=train_df, x="f_28", kde=True, color ='xkcd:lightish blue', ax=axs[3, 3])

sns.despine(top=True, right=True, left=True, bottom=True)


for ax in axs.flat:
    ax.set(ylabel='')
    ax.set_yticks([])

plt.tight_layout()

plt.show();


## === cell 5
plt.rcParams['savefig.facecolor']='white'
sns.jointplot(data=train_df,
             x="f_00",
             y='f_01',
             kind='hex',
             color="#4CB391")
plt.ylim(-3,3)
plt.xlim(-3,3)
plt.show();


## === cell 6
new_column_headers=["%02d" % x for x in range(27)]
new_column_headers = new_column_headers + ['28','29','30', 'tgt']
correlation_df = train_df.copy()
correlation_df = correlation_df.drop(labels = 'id', axis=1)
correlation_df = correlation_df.drop(labels = 'f_27', axis=1)
correlation_df.columns = new_column_headers
corr_results = correlation_df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr_results, 
                 fmt='.2f', 
                 annot = True, 
                 vmin=-1,
                 vmax=1, 
                 center= 0, 
                 cmap= 'seismic', 
                 linecolor='white', 
                 linewidth=1.5, 
                 cbar = False,
                 annot_kws={"size": 12.5})
plt.xticks(rotation=0, ha='center')
plt.yticks(rotation=0, ha='center')
plt.title('\nCorrelation Matrix (features:  f_00 thru f_30)\n\nRed: positive correlation                                         Blue: negative correlation\n\n', fontsize=15)
plt.show();


## === cell 7
new_column_headers=["%02d" % x for x in range(27)]
new_column_headers = new_column_headers + ['28','29','30', 'tgt']
correlation_df = train_df.copy()
correlation_df = correlation_df.drop(labels = 'id', axis=1)
correlation_df = correlation_df.drop(labels = 'f_27', axis=1)
plt.figure(figsize=(8,8))
c2k = ['f_07','f_08','f_09','f_10','f_11','f_12',
       'f_13','f_14','f_15','f_16','f_17','f_18']
sns.heatmap(correlation_df[c2k].corr(), 
                 fmt='.3f', 
                 annot = True, 
                 vmin=-1,
                 vmax=1, 
                 center= 0, 
                 cmap= 'seismic', 
                 linecolor='grey',
                 cbar=False,
                 linewidth=.6, 
                 annot_kws={"size": 8})
plt.yticks(rotation=0, ha='right')
plt.title('\nCorrelation Matrix (middle square)\n', fontsize=11)
plt.show();


## === cell 8
new_column_headers=["%02d" % x for x in range(27)]
new_column_headers = new_column_headers + ['28','29','30', 'tgt']
correlation_df = train_df.copy()
correlation_df = correlation_df.drop(labels = 'id', axis=1)
correlation_df = correlation_df.drop(labels = 'f_27', axis=1)
plt.figure(figsize=(7,7))
c2k = ['f_19', 'f_20','f_21','f_22','f_23','f_24','f_25','f_26']
sns.heatmap(correlation_df[c2k].corr(), 
                 fmt='.3f', 
                 annot = True, 
                 vmin=-1,
                 vmax=1, 
                 center= 0, 
                 cmap= 'seismic', 
                 linecolor='grey',
                 cbar=False,
                 linewidth=.6, 
                 annot_kws={"size": 8})
plt.yticks(rotation=0, ha='right')
plt.title('\nCorrelation Matrix (bottom right square)\n', fontsize=11)
plt.show();


## === cell 9
train_df_no_target = train_df.copy()
train_df_no_target = train_df_no_target.drop(labels = 'target', axis=1)
train_df_no_target = train_df_no_target.drop(labels = 'id', axis=1)
train_df_no_target = train_df_no_target.drop(labels = 'f_27', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_df_no_target, 
                                                   train_df['target'], 
                                                   test_size = 0.20,
                                                   random_state = 2022)


## === cell 10
params = {'n_estimators'    : 5000,
          'max_depth'       : 5,
          'learning_rate'   : 0.10,
          'random_state'    : 2022,
          'eval_metric'     : 'auc',
          'objective'       : 'binary:logistic',
          'tree_method'     : 'gpu_hist'}

xgb = XGBClassifier(**params)
xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], 
       verbose = 1000)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/3695734673.py in <cell line: 0>()
      8 
      9 xgb = XGBClassifier(**params)
---> 10 xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], 
     11        verbose = 1000)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1517             )
   1518 
-> 1519             self._Booster = train(
   1520                 params,
   1521                 train_dmatrix,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)
    179         if cb_container.before_iteration(bst, i, dtrain, evals):
    180             break
--> 181         bst.update(dtrain, i, obj)
    182         if cb_container.after_iteration(bst, i, dtrain, evals):
    183             break

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in update(self, dtrain, iteration, fobj)
   2048 
   2049         if fobj is None:
-> 2050             _check_call(
   2051                 _LIB.XGBoosterUpdateOneIter(
   2052                     self.handle, ctypes.c_int(iteration), dtrain.handle

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [01:29:46] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [01:29:46] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fba27d9af2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fba27db195a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fba27dbb3cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fba276d3c79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fba276d476c]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fba277384f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fba273d4ef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fba9ba70e2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fba9ba6d493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fba27d9af2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fba27dbb5c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fba276d3c79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fba276d476c]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fba277384f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fba273d4ef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fba9ba70e2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fba9ba6d493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7fba9ba804d8]



## === cell 11
submission_df = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
test_df = test_df.drop(['id', 'f_27'], axis=1)
submission_df['target'] = xgb.predict_proba(test_df)[:, 1]
submission_df.to_csv('submission.csv', index=False)
submission_df.head()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2873124912.py in <cell line: 0>()
      7 submission_df = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
      8 test_df = test_df.drop(['id', 'f_27'], axis=1)
----> 9 submission_df['target'] = xgb.predict_proba(test_df)[:, 1]
     10 submission_df.to_csv('submission.csv', index=False)
     11 submission_df.head()

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict_proba(self, X, validate_features, base_margin, iteration_range)
   1630             class_prob = softmax(raw_predt, axis=1)
   1631             return class_prob
-> 1632         class_probs = super().predict(
   1633             X=X,
   1634             validate_features=validate_features,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand
