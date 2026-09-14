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

geopandas==0.14.4
google-ai-generativelanguage==0.6.15
google-api-core==2.28.1
google-auth==2.38.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.2.2
google-generativeai==0.8.5
googleapis-common-protos==1.70.0
joblib==1.5.2
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
pydata-google-auth==1.9.1
python-dateutil==2.9.0.post0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
shap==0.44.1
shapely==2.1.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
xgboost==2.0.3

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

0.93782

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
The fixes address syntax errors (removing stray text and backticks), eliminate IPython‑only magics (`%matplotlib inline` and `%%time`), and prevent the accidental overwrite of the memory‑optimized dataframe. These changes let the notebook run from start to finish and correctly write a submission CSV with the required columns.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/2241619721.py", line 1
    The fixes address syntax errors (removing stray text and backticks), eliminate IPython‑only magics (`%matplotlib inline` and `%%time`), and prevent the accidental overwrite of the memory‑optimized dataframe. These changes let the notebook run from start to finish and correctly write a submission CSV with the required columns.
                                                                                          ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 1
COLAB = 'google.colab' in str(get_ipython()) 

if COLAB:        
    !pip install --q scikit-plot
    !pip install --q category_encoders
    !pip install --q shap
    !pip install --q inflection    

    from google.colab import drive
    drive.mount('/content/drive')




## === cell 2
import warnings
import random
import os
import gc
import torch
import sklearn.exceptions
import datetime
import shap




## === cell 3
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt 
import seaborn           as sns
import joblib            as jb
import xgboost           as xgb




## === cell 4
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing   import StandardScaler, MinMaxScaler, power_transform
from sklearn.preprocessing   import PowerTransformer, RobustScaler, Normalizer
from sklearn.preprocessing   import MaxAbsScaler, QuantileTransformer, LabelEncoder
from sklearn                 import metrics
from sklearn.metrics         import ConfusionMatrixDisplay, confusion_matrix




## === cell 5
from datetime                import datetime




## === cell 6
def jupyter_setting():
    
      
     
    pd.options.display.max_columns = None

    warnings.filterwarnings(action='ignore')
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category= sklearn.exceptions.UndefinedMetricWarning)

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', None)

    icecream = ["#00008b", "#960018","#008b00", "#00468b", "#8b4500", "#582c00"]
    
    colors = ["lightcoral", "sandybrown", "darkorange", "mediumseagreen",
          "lightseagreen", "cornflowerblue", "mediumpurple", "palevioletred",
          "lightskyblue", "sandybrown", "yellowgreen", "indianred",
          "lightsteelblue", "mediumorchid", "deepskyblue"]
    
    dark_red   = "#b20710"
    black      = "#221f1f"
    green      = "#009473"
    myred      = '#CD5C5C'
    myblue     = '#6495ED'
    mygreen    = '#90EE90'    
    color_cols = [myred, myblue,mygreen]
    
    return icecream, colors, color_cols

icecream, colors, color_cols = jupyter_setting()




## === cell 7
def missing_zero_values_table(df):
        mis_val         = df.isnull().sum()
        mis_val_percent = round(df.isnull().mean().mul(100), 2)
        mz_table        = pd.concat([mis_val, mis_val_percent], axis=1)
        mz_table        = mz_table.rename(columns = {df.index.name:'col_name', 
                                                     0 : 'Valores ausentes', 
                                                     1 : '% de valores totais'})
        
        mz_table['Tipo de dados'] = df.dtypes
        mz_table                  = mz_table[mz_table.iloc[:,1] != 0 ]. \
                                     sort_values('% de valores totais', ascending=False)
        
        msg = "Seu dataframe selecionado tem {} colunas e {} " + \
              "linhas. \nExistem {} colunas com valores ausentes."
            
        print (msg.format(df.shape[1], df.shape[0], mz_table.shape[0]))
        
        return mz_table.reset_index()




## === cell 8
def reduce_memory_usage(df, verbose=True):
    
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    
    for col in df.columns:
        
        col_type = df[col].dtypes
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
        
    return df




## === cell 9
def graf_label(ax, total):
    
     for i in ax.patches:
        width, height = i.get_width() -.2 , i.get_height()
        
        x, y  = i.get_xy()  
        color = 'white'
        alt   = .5
        soma  = 0 

        if height < 70:
            color = 'black'
            alt   = 1
            soma  = 10

        ax.annotate(str(round((i.get_height() * 100.0 / total), 1) )+'%', 
                    (i.get_x()+.3*width, 
                     i.get_y()+soma + alt*height),
                    color   = color,
                    weight = 'bold',
                    size   = 14)




## === cell 10
def graf_bar(df, col, title, xlabel, ylabel, tol = 0):
    
    ax     = df    
    colors = col
    
    if tol == 0: 
        total  = sum(ax)
        ax = (ax).plot(kind    ='bar',
                       stacked = True,
                       width   = .5,
                       rot     = 0,
                       color   = colors, 
                       grid    = False)
    else:
        total  = tol     
        
        ax = (ax).plot(kind    ='bar',
                       stacked = True,
                       width   = .5,
                       rot     = 0,
                       figsize = (10,6),
                       color   = colors,
                       grid    = False)

    

    title   = title #+ ' \n'
    xlabel  = '\n ' + xlabel 
    ylabel  = ylabel + ' \n'
    
    ax.set_title(title  , fontsize=22)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)    

    min = [0,23000000]
    
    graf_label(ax, total)




## === cell 11
def smape(pred, obs):
    pred = np.array(pred)
    obs = np.array(obs)   
    return 100 * np.mean(2 * abs(pred-obs)/(abs(obs)+abs(pred)))




## === cell 12
def smape_loss(y_true, y_pred):
    """SMAPE Loss"""
    return np.abs(y_true - y_pred) / (y_true + np.abs(y_pred)) * 200




## === cell 13
def SMAPE(y_true, y_pred):
    denominator = (y_true + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)




## === cell 14
def calc_erro(y, y_pred, outros=True, ruturn_score=False):
    erro   = smape(y, y_pred)    
      
    
    if outros:        
        rmse = metrics.mean_squared_error(y, y_pred, squared=False)
        mape = metrics.mean_absolute_percentage_error(y, y_pred)
        mae  = metrics.mean_absolute_error(y, y_pred)
        
        print('RMSE : {:2.5f}'.format(rmse))
        print('MAE  : {:2.5f}'.format(mae))
        print('MAPE : {:2.5f}'.format(mape))
        
        
    if ruturn_score: 
        return erro
    else: 
        print('SMAPE: {:2.5f}'.format(erro))




## === cell 15
def df_corr(df, annot_=False):
    
    df = df.corr(method ='pearson').round(5)

    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(15,12))
    ax = sns.heatmap(df, annot=annot_, mask=mask, cmap="RdBu", annot_kws={"weight": "bold", "fontsize":13})

    ax.set_title("Mapa de calor de correlação das variável", fontsize=17)

    plt.setp(ax.get_xticklabels(), 
             rotation      = 90, 
             ha            = "right",
             rotation_mode = "anchor", 
             weight        = "normal")

    plt.setp(ax.get_yticklabels(), 
             weight        = "normal",
             rotation_mode = "anchor", 
             rotation      = 0, 
             ha            = "right");




## === cell 16
def describe(df):
    var = df.columns

    ct1 = pd.DataFrame(df[var].apply(np.mean)).T
    ct2 = pd.DataFrame(df[var].apply(np.median)).T

    d1 = pd.DataFrame(df[var].apply(np.std)).T
    d2 = pd.DataFrame(df[var].apply(min)).T
    d3 = pd.DataFrame(df[var].apply(max)).T
    d4 = pd.DataFrame(df[var].apply(lambda x: x.max() - x.min())).T
    d5 = pd.DataFrame(df[var].apply(lambda x: x.skew())).T
    d6 = pd.DataFrame(df[var].apply(lambda x: x.kurtosis())).T
    d7 = pd.DataFrame(df[var].apply(lambda x: (3 *( np.mean(x) - np.median(x)) / np.std(x) ))).T

    m = pd.concat([d2, d3, d4, ct1, ct2, d1, d5, d6, d7]).T.reset_index()
    m.columns = ['attrobutes', 'min', 'max', 'range', 'mean', 'median', 'std','skew', 'kurtosis','coef_as']
    
    return m




## === cell 17
def graf_outlier(df, feature):
    col = [(0,4), (5,9)]

    df_plot = ((df[feature] - df[feature].min())/
               (df[feature].max() - df[feature].min()))

    fig, ax = plt.subplots(len(col), 1, figsize=(15,7))

    for i, (x) in enumerate(col): 
        sns.boxplot(data = df_plot.iloc[:, x[0]:x[1] ], ax = ax[i]); 




## === cell 18
def diff(t_a, t_b):
    from dateutil.relativedelta import relativedelta
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)




## === cell 19
def free_gpu_cache():
    
    gc.collect()
    torch.cuda.empty_cache()




## === cell 20
def graf_eval():

    results     = model.evals_result()
    ntree_limit = model.best_ntree_limit

    plt.figure(figsize=(20,7))

    for i, error in  enumerate(['mlogloss', 'merror']):#
        
        plt.subplot(1,2,i+1)
        plt.plot(results["validation_0"][error], label="Treinamento")
        plt.plot(results["validation_1"][error], label="Validação")

        plt.axvline(ntree_limit, 
                    color="gray", 
                    label="N. de árvore ideal {}".format(ntree_limit))
                    
        
        title_name ='\n' + error.upper() + ' PLOT \n'
        plt.title(title_name)
        plt.xlabel("Número de árvores")
        plt.ylabel(error)
        plt.legend();




## === cell 21
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)




## === cell 22
def linear_fit_slope(y):
    """Return the slope of a linear fit to a series."""
    y_pure = y.dropna()
    length = len(y_pure)
    x = np.arange(0, length)
    slope, intercept = np.polyfit(x, y_pure.values, deg=1)
    return slope




## === cell 23
def linear_fit_intercept(y):
    """Return the intercept of a linear fit to a series."""
    y_pure = y.dropna()
    length = len(y_pure)
    x = np.arange(0, length)
    slope, intercept = np.polyfit(x, y_pure.values, deg=1)
    return intercept




## === cell 24
def cromer_v(x, y):
    cm       = pd.crosstab(x, y).to_numpy()        
    n        = cm.sum()
    r, k     = cm.shape
    chi2     = stats.chi2_contingency(cm)[0]
    chi2corr = max(0, chi2 - (k-1) * (r-1) /(n-1))
    kcorr    = k - (k-1) **2/(n-1)
    rcorr    = r - (r-1) **2/(n-1)    
    v        = np.sqrt((chi2corr/n) / (min(kcorr-1, rcorr-1)))        
    return v  




## === cell 25
def generate_category_table(data):

    cols    = data.select_dtypes(include='object').columns
    dataset = pd.DataFrame()

    for i in cols:
        corr = []
        for x in cols: 
            corr.append(cromer_v(data[i],data[x]))

        aux     = pd.DataFrame({i:corr})
        dataset = pd.concat([dataset, aux], axis=1) 

    return dataset.set_index(dataset.columns)




## === cell 26
def graf_feature_corr(df, annot_=False, threshold=.8, print_var=False):
    """
    Plot a correlation heatmap using only numeric columns.
    """
    df_numeric = df.select_dtypes(include=[np.number])
    df_corr = df_numeric.corr(method='pearson').round(5)

    mask = np.zeros_like(df_corr)
    mask[np.triu_indices_from(mask)] = True

    ax = sns.heatmap(df_corr, annot=annot_, mask=mask, cmap="RdBu", 
                     annot_kws={"weight": "bold", "fontsize":13})

    ax.set_title("Mapa de calor de correlação das variável", fontsize=17)

    plt.setp(ax.get_xticklabels(), 
             rotation=90, ha="right", rotation_mode="anchor", weight="normal")
    plt.setp(ax.get_yticklabels(), 
             weight="normal", rotation_mode="anchor", rotation=0, ha="right")
    
    if print_var: 
        print('Variáveis autocorrelacionadas threshold={:2.2f}'.format(threshold))
        df_corr_sel = df_corr[abs(df_corr)>threshold][df_corr!=1.0].unstack().dropna().reset_index()
        df_corr_sel.columns =  ['var_1', 'var_2', 'corr']
        display(df_corr_sel)




## === cell 27
def plot_roc_curve(fpr, tpr, label=None):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, "r-", label=label)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for FLAI 08')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc="lower right")
    plt.grid(True)




## === cell 28
def feature_engineering(df_):
    
    var_f27 = ''
    for col in df_['f_27']: 
        var_f27 +=col

    var_f27 = list(set(var_f27))
    var_f27.sort()
    
    df_["fe_f_27_unique"] = df_["f_27"].apply(lambda x: len(set(x)))
    
    for letra in var_f27:             
        df_['fe_' + letra.lower() + '_count'] = df2_train["f_27"].str.count(letra)
        
    return df_ 




## === cell 29
paths = ['img', 'Data', 'Data/pkl', 'Data/submission', 'Data/tunning', 
         'model', 'model/preds', 'model/optuna','model/preds/test', 
         'model/preds/test/n1', 'model/preds/test/n2', 'model/preds/test/n3', 
         'model/preds/train', 'model/preds/train/n1', 'model/preds/train/n2', 
         'model/preds/train/n3', 'model/preds/param']

for path in paths:
    try:
        os.mkdir(path)       
    except:
        pass  




## === cell 30
path        = '/content/drive/MyDrive/kaggle/Tabular Playground Series/05 - Maio/' if COLAB else ''   
path        = '../input/tabular-playground-series-may-2022/'
path_data   = ''  
target      = 'target'
path_automl = 'automl/'




## === cell 31
df1_train     = pd.read_csv(path + path_data + 'train.csv')
df1_test      = pd.read_csv(path + path_data + 'test.csv')
df_submission = pd.read_csv(path + path_data + 'sample_submission.csv')

df1_train.shape, df1_test.shape, df_submission.shape




## === cell 32
df1_train.head()




## === cell 33
df1_test.head()




## === cell 34
df2_train = reduce_memory_usage(df1_train)
df2_test  = reduce_memory_usage(df1_test)

float16_cols = df2_train.select_dtypes(include=['float16']).columns
df2_train[float16_cols] = df2_train[float16_cols].astype(np.float32)
float16_cols_test = df2_test.select_dtypes(include=['float16']).columns
df2_test[float16_cols_test] = df2_test[float16_cols_test].astype(np.float32)




## === cell 36
print('TREINO')
print('Number of Rows: {}'.format(df2_train.shape[0]))
print('Number of Columns: {}'.format(df2_train.shape[1]), end='\n\n')

print('TESTE')
print('Number of Rows: {}'.format(df2_test.shape[0]))
print('Number of Columns: {}'.format(df2_test.shape[1]))




## === cell 37
df2_train.info()




## === cell 38
df2_test.info()




## === cell 39
print(f'{3*"="} For Pandas {10*"="}\n{(df2_train.dtypes).value_counts()}')
print(f'\n{3*"="} For Datatable {7*"="}\n{(df2_test.dtypes).value_counts()}')




## === cell 40
for col in df2_train.select_dtypes(np.int8).columns.drop(target):   
    num = df2_train[col].unique().tolist()
    num.sort()
    print('-'* 70)
    print(f'{col} unique: {num}')
    print('-'* 70)
    print()




## === cell 41
for col in df2_train.select_dtypes(np.int8).columns.drop(target):   
    df2_train[col] = df2_train[col].astype(object)   
    df2_test[col]  = df2_test[col].astype(object)    




## === cell 42
df2_train.info()




## === cell 43
df2_test.info()




## === cell 44
missing = missing_zero_values_table(df2_train)
missing[:].style.background_gradient(cmap='Reds')




## === cell 45
missing = missing_zero_values_table(df2_test)
missing[:].style.background_gradient(cmap='Reds')




## === cell 46
feature_float = df2_test.select_dtypes(np.number).columns.to_list()
feature_cat   = df2_test.select_dtypes(object).columns.to_list()

feature_float.remove('id')

msg = 'Temos {} variávies numéricas e {} categóricas.'
print(msg.format(len(feature_float), len(feature_cat)))




## === cell 47
df2_train.drop([target, 'id'], axis=1).describe().T.style.background_gradient(cmap='YlOrRd')




## === cell 48
fig, ax = plt.subplots(figsize=(5, 5))

pie = ax.pie([len(df2_train), len(df2_test)],
             labels   = ["Train dataset", "Test dataset"],
             textprops= {"fontsize": 15},
             autopct  = '%1.1f%%')

ax.axis("equal")
ax.set_title("Comparação de comprimento do conjunto de dados \n", fontsize=18)
fig.set_facecolor('white')
plt.show()




## === cell 49
lines   = int(len(feature_float)/2)
fig, ax = plt.subplots(lines,2 ,figsize=(20,20))

for i,feature in enumerate(feature_float):
    plt.subplot(lines,2,i+1)
    sns.histplot(data=df2_train, x=df2_train[feature].astype(np.float64),color='blue', alpha=0.5, label='train', bins=1000)
    sns.histplot(data=df2_test , x=df2_test[feature].astype(np.float64) ,color='teal', alpha=0.5, label='test' , bins=1000)     
    plt.xlabel(feature, fontsize=12)
    plt.legend()
         
plt.suptitle('DistPlot: train & test data', fontsize=20)




## === cell 50
fig, ax = plt.subplots(figsize=(5, 5))

plt.pie([ len(feature_cat), len(feature_float)], 
        labels=['Categorical', 'Continuos' ],
        textprops={'fontsize': 13},
        autopct='%1.1f%%')

ax.set_title("Comparação variáveis continuas/categóricas \n Dataset Treino/Teste", fontsize=18)
fig.set_facecolor('white')
plt.show()




## === cell 51
plt.figure(figsize=(7,5))    

graf_bar(df2_train.groupby([target])[target].count() , 
         icecream, 
         'Distribuição da variável preditora', 
         target, 
         'Quantidade pessoas')




## === cell 52
for i in feature_cat:
    print("Coluna: ",i)
    print(df2_train[[i]].value_counts(), "\n")




## === cell 53
feature_cat.remove('f_27')
print(feature_cat)




## === cell 54
plt.figure(figsize=(20,40))

for i, col in enumerate(feature_cat):
    plt.subplot(int(len(feature_cat)/2) + 1,2,i+1)
    ax = sns.countplot(data=df1_train, y=col, hue=target)    




## === cell 55
plt.figure(figsize=(20,40))
graf_feature_corr(df=df2_train.copy().drop('id', axis=1), annot_=False, threshold=.5, print_var=False)




## === cell 56
plt.subplots(figsize=(20, 15))

for i, col in enumerate(feature_float):    
    plt.subplot(int(len(feature_float)/3 +1),3,i+1)
    sns.kdeplot(data=df2_train, x=col, hue=target, legend=True, shade=True, multiple='stack');  




## === cell 57
f, ax = plt.subplots(figsize=(20, 20))

for i, col in enumerate(feature_float): 
    plt.subplot(int(len(feature_float)/3 +1),3,i+1)
    sns.boxplot(data=df2_train, x=target, y=col)    





## === cell 58
df3_train = df2_train.copy()
df3_test  = df2_test.copy()

df3_train.drop(['f_27'], axis=1 , inplace=True)
df3_test.drop(['f_27'], axis=1 , inplace=True)

for col in df3_train.select_dtypes(object).columns: 
    df3_train[col] = df3_train[col].astype(np.int32)
    df3_test[col] = df3_test[col].astype(np.int32)    




## === cell 59
X      = df3_train.drop([target, 'id'], axis=1)
y      = df3_train[target]
X_test = df3_test.drop(['id'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, 
    test_size    = 0.3,
    shuffle      = True, 
    stratify     = y, 
    random_state = 12359)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape , X_test.shape




## === cell 60
X_valid, X_test_final, y_valid, y_test_final = train_test_split(
    X_valid, 
    y_valid, 
    test_size    = 0.3,
    shuffle      = True, 
    stratify     = y_valid, 
    random_state = 12359)

X_valid.shape, y_valid.shape, X_test_final.shape, y_test_final.shape




## === cell 61
seed   = 12359
params = {'objective'        : 'binary:logistic',   
          'eval_metric'      : 'auc',  
          'n_estimators'     : 1000,          
          'random_state'     : seed}

if torch.cuda.is_available():           
    params.update({'tree_method': 'gpu_hist','predictor': 'gpu_predictor'})
    
params




## === cell 62
scalers = [StandardScaler(),
           RobustScaler(), 
           MinMaxScaler(), 
           MaxAbsScaler(),            
           QuantileTransformer(output_distribution='normal', random_state=0)]

model_baseline = xgb.XGBClassifier(**params)
scaler_best    = None
model_best     = None
cols           = X_test.columns
f1_best        = 0 
auc_best       = 0 

for scaler in scalers: 
    
    X_train_s = X_train.copy() 
    X_valid_s = X_valid.copy()
    
    if scaler is not None:                      
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train_s), columns=cols)
        X_valid_s = pd.DataFrame(scaler.transform(X_valid_s), columns=cols)
        
    model_baseline.fit(X_train_s, y_train, verbose=False)

    y_pred_prob_tr = model_baseline.predict_proba(X_train_s)[:,1]
    y_pred_prob_vl = model_baseline.predict_proba(X_valid_s)[:,1]
    
    y_pred_tr = (y_pred_prob_tr>.5).astype(int) 
    y_pred_vl = (y_pred_prob_vl>.5).astype(int)

    f1     = metrics.f1_score(y_valid, y_pred_vl)
    auc_vl = metrics.roc_auc_score(y_valid, y_pred_prob_vl)
    auc_tr = metrics.roc_auc_score(y_train, y_pred_prob_tr)
        
    print('AUC Trn: {:2.5f} - AUC Val: {:2.5f} - F1: {:2.5f} => {}'.format(auc_tr, auc_vl, f1, scaler))
    
    if auc_vl>auc_best:
        f1_best     = f1    
        auc_best    = auc_vl
        scaler_best = scaler
        model_best  = model_baseline
        
    del scaler, f1, auc_vl
    
print()
print('The Best')  
print('Scaler: {}'.format(scaler_best))    
print('AUC   : {:2.5f}'.format(auc_best))
print()




## === cell 63
features = pd.Series(model_best.feature_importances_)
features.index = cols

features.sort_values(ascending=True, inplace=True)
features.plot(kind='barh', figsize=(15,15))




## === cell 64
path=''




## === cell 65
X_test_final_sc = pd.DataFrame(scaler_best.transform(X_test_final), columns=cols)
y_pred_prob     = model_best.predict_proba(X_test_final_sc)[:,1]
auc             = metrics.roc_auc_score(y_test_final, y_pred_prob)

print('AUC dados não visto : {:2.5f}'.format(auc))

X_test_sc = pd.DataFrame(scaler_best.transform(X_test), columns=cols)
y_pred_ts = model_best.predict_proba(X_test_sc)[:,1]

df_submission[target] = y_pred_ts
df_submission.to_csv(path + 'Data/submission/xgb_base_line_score_01_{:2.5f}.csv'.format(auc), index=False)




## === cell 66
pass




## === cell 67
pass




## === cell 68
pass




## === cell 69
pass




## === cell 70
pass




## === cell 71
pass




## === cell 72
pass




## === cell 73
pass




## === cell 74
pass
```

## --- ERROR in cell 74, traceback:
  File "/tmp/ipykernel_55/1705845959.py", line 2
    ```
    ^
SyntaxError: invalid syntax
