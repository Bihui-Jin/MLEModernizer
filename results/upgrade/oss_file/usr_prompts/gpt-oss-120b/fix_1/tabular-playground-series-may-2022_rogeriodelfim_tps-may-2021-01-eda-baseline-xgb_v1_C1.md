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
COLAB = 'google.colab' in str(get_ipython()) 

if COLAB:        
    !pip install --q scikit-plot
    !pip install --q category_encoders
    !pip install --q shap
    !pip install --q inflection    

    from google.colab import drive
    drive.mount('/content/drive')


## === cell 1
import warnings
import random
import os
import gc
import torch
import sklearn.exceptions
import datetime
import shap


## === cell 2
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt 
import seaborn           as sns
import joblib            as jb
import xgboost           as xgb


## === cell 3
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing   import StandardScaler, MinMaxScaler, power_transform
from sklearn.preprocessing   import PowerTransformer, RobustScaler, Normalizer
from sklearn.preprocessing   import MaxAbsScaler, QuantileTransformer, LabelEncoder
from sklearn                 import metrics
from sklearn.metrics         import ConfusionMatrixDisplay, confusion_matrix


## === cell 4
from datetime                import datetime


## === cell 5
def jupyter_setting():
    
    %matplotlib inline
      
     
    pd.options.display.max_columns = None

    warnings.filterwarnings(action='ignore')
    warnings.simplefilter('ignore')
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


## === cell 6
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


## === cell 7
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
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
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


## === cell 8
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


## === cell 9
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


## === cell 10
def smape(pred, obs):
    pred = np.array(pred)
    obs = np.array(obs)   
    return 100 * np.mean(2 * abs(pred-obs)/(abs(obs)+abs(pred)))


## === cell 11
def smape_loss(y_true, y_pred):
    """SMAPE Loss"""
    return np.abs(y_true - y_pred) / (y_true + np.abs(y_pred)) * 200


## === cell 12
def SMAPE(y_true, y_pred):
    denominator = (y_true + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


## === cell 13
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


## === cell 14
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


## === cell 15
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


## === cell 16
def graf_outlier(df, feature):
    col = [(0,4), (5,9)]

    df_plot = ((df[feature] - df[feature].min())/
               (df[feature].max() - df[feature].min()))

    fig, ax = plt.subplots(len(col), 1, figsize=(15,7))

    for i, (x) in enumerate(col): 
        sns.boxplot(data = df_plot.iloc[:, x[0]:x[1] ], ax = ax[i]); 


## === cell 17
def diff(t_a, t_b):
    from dateutil.relativedelta import relativedelta
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


## === cell 18
def free_gpu_cache():
    

    
    gc.collect()
    torch.cuda.empty_cache()


## === cell 19
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


## === cell 20
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)


## === cell 21
def linear_fit_slope(y):
    """Return the slope of a linear fit to a series."""
    y_pure = y.dropna()
    length = len(y_pure)
    x = np.arange(0, length)
    slope, intercept = np.polyfit(x, y_pure.values, deg=1)
    return slope


## === cell 22
def linear_fit_intercept(y):
    """Return the intercept of a linear fit to a series."""
    y_pure = y.dropna()
    length = len(y_pure)
    x = np.arange(0, length)
    slope, intercept = np.polyfit(x, y_pure.values, deg=1)
    return intercept


## === cell 23
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


## === cell 24
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


## === cell 25
def graf_feature_corr(df, annot_=False, threshold=.8, print_var=False):
    
    df = df.corr(method ='pearson').round(5)

    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True

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
    
    if print_var: 
        print('Variáveis autocorrelacionadas threshold={:2.2f}'.format(threshold))
        df_corr = df[abs(df)>threshold][df!=1.0].unstack().dropna().reset_index()
        df_corr.columns =  ['var_1', 'var_2', 'corr']
        display(df_corr)


## === cell 26
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


## === cell 27
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


## === cell 28
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


## === cell 29
path        = '/content/drive/MyDrive/kaggle/Tabular Playground Series/05 - Maio/' if COLAB else ''   
path        = '../input/tabular-playground-series-may-2022/'
path_data   = ''  
target      = 'target'
path_automl = 'automl/'


## === cell 30
df1_train     = pd.read_csv(path + path_data + 'train.csv')
df1_test      = pd.read_csv(path + path_data + 'test.csv')
df_submission = pd.read_csv(path + path_data + 'sample_submission.csv')

df1_train.shape, df1_test.shape, df_submission.shape


## === cell 31
df1_train.head()


## === cell 32
df1_test.head()


## === cell 33
df1_train = reduce_memory_usage(df1_train)
df1_test  = reduce_memory_usage(df1_test)


## === cell 34
df2_train = df1_train.copy()
df2_test  = df1_test.copy()


## === cell 35
print('TREINO')
print('Number of Rows: {}'.format(df2_train.shape[0]))
print('Number of Columns: {}'.format(df2_train.shape[1]), end='\n\n')

print('TESTE')
print('Number of Rows: {}'.format(df2_test.shape[0]))
print('Number of Columns: {}'.format(df2_test.shape[1]))


## === cell 36
df2_train.info()


## === cell 37
df2_test.info()


## === cell 38
print(f'{3*"="} For Pandas {10*"="}\n{(df2_train.dtypes).value_counts()}')
print(f'\n{3*"="} For Datatable {7*"="}\n{(df2_test.dtypes).value_counts()}')


## === cell 39
for col in df2_train.select_dtypes(np.int8).columns.drop(target):   
    num = df2_train[col].unique().tolist()
    num.sort()
    print('-'* 70)
    print('{} unique: {}'.format(col, num))    
    print('-'* 70)
    print()


## === cell 40
for col in df2_train.select_dtypes(np.int8).columns.drop(target):   
    df2_train[col] = df2_train[col].astype(object)   
    df2_test[col]  = df2_test[col].astype(object)    


## === cell 41
df2_train.info()


## === cell 42
df2_test.info()


## === cell 43
missing = missing_zero_values_table(df2_train)
missing[:].style.background_gradient(cmap='Reds')


## === cell 44
missing = missing_zero_values_table(df2_test)
missing[:].style.background_gradient(cmap='Reds')


## === cell 45
feature_float = df2_test.select_dtypes(np.number).columns.to_list()
feature_cat   = df2_test.select_dtypes(object).columns.to_list()

feature_float.remove('id')

msg = 'Temos {} variávies numéricas e {} categóricas.'
print(msg.format(len(feature_float), len(feature_cat)))


## === cell 46
df2_train.drop([target, 'id'], axis=1).describe().T.style.background_gradient(cmap='YlOrRd')


## === cell 47
fig, ax = plt.subplots(figsize=(5, 5))

pie = ax.pie([len(df2_train), len(df2_test)],
             labels   = ["Train dataset", "Test dataset"],
             textprops= {"fontsize": 15},
             autopct  = '%1.1f%%')

ax.axis("equal")
ax.set_title("Comparação de comprimento do conjunto de dados \n", fontsize=18)
fig.set_facecolor('white')
plt.show();


## === cell 48
lines   = int(len(feature_float)/2)
fig, ax = plt.subplots(lines,2 ,figsize=(20,20))

for i,feature in enumerate(feature_float):
    plt.subplot(lines,2,i+1)
    sns.histplot(data=df2_train, x=df2_train[feature],color='blue', alpha=0.5, label='train', bins=1000)
    sns.histplot(data=df2_test , x=df2_test[feature] ,color='teal', alpha=0.5, label='test' , bins=1000)     
    plt.xlabel(feature, fontsize=12)
    plt.legend()
         
plt.suptitle('DistPlot: train & test data', fontsize=20);


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/67680167.py in <cell line: 0>()
      4 for i,feature in enumerate(feature_float):
      5     plt.subplot(lines,2,i+1)
----> 6     sns.histplot(data=df2_train, x=df2_train[feature],color='blue', alpha=0.5, label='train', bins=1000)
      7     sns.histplot(data=df2_test , x=df2_test[feature] ,color='teal', alpha=0.5, label='test' , bins=1000)
      8     plt.xlabel(feature, fontsize=12)

/usr/local/lib/python3.11/dist-packages/seaborn/distributions.py in histplot(data, x, y, hue, weights, stat, bins, binwidth, binrange, discrete, cumulative, common_bins, common_norm, multiple, element, fill, shrink, kde, kde_kws, line_kws, thresh, pthresh, pmax, cbar, cbar_ax, cbar_kws, palette, hue_order, hue_norm, color, log_scale, legend, ax, **kwargs)
   1430     if p.univariate:
   1431 
-> 1432         p.plot_univariate_histogram(
   1433             multiple=multiple,
   1434             element=element,

/usr/local/lib/python3.11/dist-packages/seaborn/distributions.py in plot_univariate_histogram(self, multiple, element, fill, common_norm, common_bins, shrink, kde, kde_kws, color, legend, line_kws, estimate_kws, **plot_kws)
    497             widths *= shrink
    498             index = pd.MultiIndex.from_arrays([
--> 499                 pd.Index(edges, name="edges"),
    500                 pd.Index(widths, name="widths"),
    501             ])

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in __new__(cls, data, dtype, copy, name, tupleize_cols)
    574         klass = cls._dtype_to_subclass(arr.dtype)
    575 
--> 576         arr = klass._ensure_array(arr, arr.dtype, copy=False)
    577         result = klass._simple_new(arr, name, refs=refs)
    578         if dtype is None and is_pandas_object and data_dtype == np.object_:

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _ensure_array(cls, data, dtype, copy)
    599         elif dtype == np.float16:
    600             # float16 not supported (no indexing engine)
--> 601             raise NotImplementedError("float16 indexes are not supported")
    602 
    603         if copy:

NotImplementedError: float16 indexes are not supported

## === cell 49
fig, ax = plt.subplots(figsize=(5, 5))

plt.pie([ len(feature_cat), len(feature_float)], 
        labels=['Categorical', 'Continuos' ],
        textprops={'fontsize': 13},
        autopct='%1.1f%%')

ax.set_title("Comparação variáveis continuas/categóricas \n Dataset Treino/Teste", fontsize=18)
fig.set_facecolor('white')
plt.show()


## === cell 50
plt.figure(figsize=(7,5))    

graf_bar(df2_train.groupby([target])[target].count() , 
         icecream, 
         'Distribuição da variável preditora', 
         target, 
         'Quantidade pessoas');


## === cell 51
for i in feature_cat:
    print("Coluna: ",i)
    print(df2_train[[i]].value_counts(), "\n")


## === cell 52
feature_cat.remove('f_27')
print(feature_cat)


## === cell 53
plt.figure(figsize=(20,40))

for i, col in enumerate(feature_cat):
    plt.subplot(int(len(feature_cat)/2) + 1,2,i+1)
    ax = sns.countplot(data=df1_train, y=col, hue=target)    


## === cell 54
plt.figure(figsize=(20,15))
graf_feature_corr(df=df2_train.copy().drop('id', axis=1), annot_=False, threshold=.5, print_var=False)


## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3659953496.py in <cell line: 0>()
      1 plt.figure(figsize=(20,15))
----> 2 graf_feature_corr(df=df2_train.copy().drop('id', axis=1), annot_=False, threshold=.5, print_var=False)

/tmp/ipykernel_11/1695127873.py in graf_feature_corr(df, annot_, threshold, print_var)
      1 def graf_feature_corr(df, annot_=False, threshold=.8, print_var=False):
      2 
----> 3     df = df.corr(method ='pearson').round(5)
      4 
      5     # Máscara para ocultar a parte superior direita do gráfico, pois é uma duplicata

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: 'ACBADABECB'

## === cell 55
plt.subplots(figsize=(20, 15))

for i, col in enumerate(feature_float):    
    plt.subplot(int(len(feature_float)/3 +1),3,i+1)
    sns.kdeplot(data=df2_train, x=col, hue=target, legend=True, shade=True, multiple='stack');  


## === cell 56
f, ax = plt.subplots(figsize=(20, 20))

for i, col in enumerate(feature_float): 
    plt.subplot(int(len(feature_float)/3 +1),3,i+1)
    sns.boxplot(data=df2_train, x=target, y=col)    


## === cell 57
df3_train = df2_train.copy()
df3_test  = df2_test.copy()

df3_train.drop(['f_27'], axis=1 , inplace=True)
df3_test.drop(['f_27'], axis=1 , inplace=True)

for col in df3_train.select_dtypes(object).columns: 
    df3_train[col] = df3_train[col].astype(np.int32)
    df3_test[col] = df3_test[col].astype(np.int32)    


## === cell 58
X      = df3_train.drop([target, 'id'], axis=1)
y      = df3_train[target]
X_test = df3_test.drop(['id'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      test_size    = 0.3,
                                                      shuffle      = True, 
                                                      stratify     = y, 
                                                      random_state = 12359)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape , X_test.shape


## === cell 59
X_valid, X_test_final, y_valid, y_test_final = train_test_split(X_valid, 
                                                                y_valid, 
                                                                test_size    = 0.3,
                                                                shuffle      = True, 
                                                                stratify     = y_valid, 
                                                                random_state = 12359)

X_valid.shape, y_valid.shape, X_test_final.shape, y_test_final.shape


## === cell 60
seed   = 12359
params = {'objective'        : 'binary:logistic',   
          'eval_metric'      : 'auc',  
          'n_estimators'     : 1000,          
          'random_state'     : seed}

if torch.cuda.is_available():           
    params.update({'tree_method': 'gpu_hist','predictor': 'gpu_predictor'})
    
params


## === cell 61
%%time 

scalers = [StandardScaler(),
           RobustScaler(), 
           MinMaxScaler(), 
           MaxAbsScaler(),            
           QuantileTransformer(output_distribution='normal', random_state=0), 
           
           ]

model_baseline = xgb.XGBClassifier(**params)
scaler_best    = None
model_best     = None
cols           = X_test.columns
f1_best        = 0 
auc_best       = 0 

for scaler in scalers: 
    
    X_train_s = X_train.copy() 
    X_valid_s = X_valid.copy()
    
    if scaler!=None:                      
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


## === cell 62
features       = pd.Series(model_baseline.feature_importances_)
features.index = cols

features.sort_values(ascending=True, inplace=True)
features.plot(kind ='barh', figsize=(15,15));


## === cell 63
path=''


## === cell 64
X_test_final_sc = pd.DataFrame(scaler_best.transform(X_test_final), columns=cols)
y_pred_prob     = model_best.predict_proba(X_test_final_sc)[:,1]
auc             = metrics.roc_auc_score(y_test_final, y_pred_prob)

print('AUC dados não visto : {:2.5f}'.format(auc))

X_test_sc = pd.DataFrame(scaler_best.transform(X_test), columns=cols)
y_pred_ts = model_best.predict_proba(X_test_sc)[:,1]

df_submission[target] = y_pred_ts
df_submission.to_csv(path + 'Data/submission/xgb_base_line_score_01_{:2.5f}.csv'.format(auc), index=False)


## === cell 65
def cross_val_model(model_, model_name_, X_, y_, X_test_, target_, scalers_, lb_, fold_=5, path_='',  
                    seed_=12359, feature_scaler_=None, print_report_=False, save_submission_=False):
    
    n_estimators = model_.get_params()['n_estimators']
             
    valid_preds     = {}
    taco            = 76 
    acc_best        = 0
    df_proba        = pd.DataFrame()
    feature_imp     = pd.DataFrame()
    test_preds      = []
    test_pred_proba = np.zeros((1, 1))
    preds           = []
    model           = []
    
    for i, scaler_ in enumerate(scalers_): 

        time_start = datetime.now()
        score      = []        
                
        if scaler_!=None:            
            string_scaler = str(scaler_)        
            string_scaler = string_scaler[:string_scaler.index('(')]
        else:
            string_scaler = None 
            
        y_pred_test = np.zeros(len(X_test_))

        folds = KFold(n_splits=fold_, shuffle=True, random_state=seed_)
        folds = StratifiedKFold(n_splits=fold_, shuffle=True, random_state=seed_)
        
        print('='*taco)
        print('Scaler: {} - n_estimators: {}'.format(string_scaler, n_estimators))
        print('='*taco)
        
        pred_test=0 
        
        for fold, (trn_idx, val_idx) in enumerate(folds.split(X_, y_, groups=y_)): 

            time_fold_start = datetime.now()

            X_trn, X_val = X_.iloc[trn_idx], X_.iloc[val_idx]
            y_trn, y_val = y_.iloc[trn_idx], y_.iloc[val_idx] 
            
            if scaler_!=None: 
                X_tst = X_test_.copy()
                if feature_scaler_!=None:                     
                    X_trn[feature_scaler_] = scaler_.fit_transform(X_trn[feature_scaler_])
                    X_val[feature_scaler_] = scaler_.transform(X_val[feature_scaler_])                      
                    X_tst[feature_scaler_] = scaler_.transform(X_tst[feature_scaler_])
                else:            
                    X_trn = scaler_.fit_transform(X_trn)
                    X_val = scaler_.transform(X_val)
                    X_tst = scaler_.transform(X_test_.copy())
                
            model_.fit(X_trn, y_trn,
                       eval_set              = [(X_trn, y_trn), (X_val, y_val)],          
                       early_stopping_rounds = int(n_estimators*.1),
                       verbose               = False)
            
            y_pred_val_prob = model_.predict_proba(X_val, ntree_limit=model_.best_ntree_limit)[:,1]    
            y_pred_val      = (y_pred_val_prob>.5).astype(int)

            preds.append(model_.predict(X_tst))    
            
            pred_test += model_.predict_proba(X_tst)[:, 1] / folds.n_splits
                
            df_prob_temp    = pd.DataFrame(y_pred_val_prob)

            df_prob_temp['fold']    = fold+1
            df_prob_temp['id']      = val_idx            
            df_prob_temp['y_val']   = y_val.values
            df_prob_temp['y_pred']  = y_pred_val            
            df_prob_temp['y_proba'] = y_pred_val_prob
            df_prob_temp['scaler']  = str(string_scaler)
                        
            acc   = metrics.accuracy_score(y_val, y_pred_val)
            auc   = metrics.roc_auc_score(y_val, y_pred_val_prob)
            f1    = metrics.f1_score(y_val, y_pred_val) 
            prec  = metrics.log_loss (y_val, y_pred_val)
            
            score.append(auc)     
            
            feat_imp = pd.DataFrame(index   = X_.columns,
                                    data    = model_.feature_importances_,
                                    columns = ['fold_{}'.format(fold+1)])

            feat_imp['auc_'+str(fold+1)] = auc
            feature_imp = pd.concat([feature_imp, feat_imp], axis=1)
            
            time_fold_end = diff(time_fold_start, datetime.now())
            msg = '[Fold {}] AUC: {:2.5f} - F1-score: {:2.5f} - L. Loss: {:2.5f}  - {}'
            print(msg.format(fold+1, auc, f1, prec, time_fold_end))
            
            dic_model = {'scaler' : scaler_, 
                         'fold'   : fold+1, 
                         'model'  : model_ }
            
            model.append(dic_model)

        score_mean = np.mean(score) 
        score_std  = np.std(score)

        if score_mean > acc_best:     
            acc_best    = score_mean           
            model_best  = model_    
            scaler_best = scaler_

        time_end = diff(time_start, datetime.now())   

        print('-'*taco)
        print('[Mean Fold] AUC: {:2.5f} std: {:2.5f} - {}'.format(score_mean, score_std, time_end))
        print('='*taco)
        print()
               
        if save_submission_:
            name_file_sub  = model_name_ + '_' + str(i+1) + '_' + str(scaler_).lower()[:4] + '.csv'
            name_file_sub  = path_ + 'Data/submission/' + name_file_sub.format(score_mean)        
            df_sub         = df_submission.copy()
            df_sub[target] = pred_test
            df_sub.to_csv(name_file_sub, index=False)
            
        if print_report_:
            y_pred = df_prob_temp[df_prob_temp['scaler']==str(string_scaler)]['y_pred']
            y_vl   = df_prob_temp[df_prob_temp['scaler']==str(string_scaler)]['y_val']
            print(metrics.classification_report(y_vl,y_pred))

    print('-'*taco)
    print('Scaler Best: {}'.format(scaler_best))
    print('Score      : {:2.5f}'.format(acc_best))
    print('-'*taco)
    print()

    return model, df_prob_temp.sort_values(by=['scaler','id']) , feature_imp 


## === cell 66
%%time 

seed        = 12359
eval_metric = ['auc', 'error']                 
scalers     = [QuantileTransformer(output_distribution='normal', random_state=0)]

params = {'objective'        : 'binary:logistic',   
          'eval_metric'      : eval_metric,  
          'n_estimators'     : 1000,                        
          'random_state'     : seed}

if torch.cuda.is_available():           
    params.update({'tree_method' : 'gpu_hist',                    
                   'predictor'   : 'gpu_predictor'})

model, df_proba, feature_imp = \
    cross_val_model(model_           = xgb.XGBClassifier(**params),
                    model_name_      = 'xgb_bs_vc_score_02_{:2.5f}',
                    X_               = X,
                    y_               = y,
                    X_test_          = X_test,
                    target_          = target,
                    scalers_         = scalers,
                    fold_            = 5, 
                    lb_              = None,
                    path_            = path,
                    seed_            = seed, 
                    feature_scaler_  = None, 
                    print_report_    = True, 
                    save_submission_ = True)
print()


## --- ERROR in cell 66, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<timed exec> in <module>

/tmp/ipykernel_11/3831920632.py in cross_val_model(model_, model_name_, X_, y_, X_test_, target_, scalers_, lb_, fold_, path_, seed_, feature_scaler_, print_report_, save_submission_)
     72             # Predição
     73             # ----------------------------------------------------
---> 74             y_pred_val_prob = model_.predict_proba(X_val, ntree_limit=model_.best_ntree_limit)[:,1]
     75             y_pred_val      = (y_pred_val_prob>.5).astype(int)
     76 

AttributeError: 'XGBClassifier' object has no attribute 'best_ntree_limit'

## === cell 67
plt.figure(figsize=(15,12))
for fold, col in enumerate(feature_imp.filter(regex=r'fold').columns):            
    col_acc = 'auc_' + str(fold+1)
    df_fi = feature_imp.sort_values(by=col, ascending=False).reset_index().iloc[:15]
    df_fi = df_fi[['index', col, col_acc]]
    df_fi.columns = ['Feature', 'score', col_acc]
    plt.subplot(3,2, fold+1)
    sns.barplot(x='score', y='Feature', data=df_fi)    
    plt.title('Fold {} - score: {:2.5f}'.format(fold+1, df_fi[col_acc].mean()), 
              fontdict={'fontsize':18})    

plt.suptitle('Feature Importance XGB', y=1.05, fontsize=24);
plt.tight_layout(h_pad=3.0); 


## --- ERROR in cell 67, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2558928975.py in <cell line: 0>()
      1 plt.figure(figsize=(15,12))
----> 2 for fold, col in enumerate(feature_imp.filter(regex=r'fold').columns):
      3     col_acc = 'auc_' + str(fold+1)
      4     df_fi = feature_imp.sort_values(by=col, ascending=False).reset_index().iloc[:15]
      5     df_fi = df_fi[['index', col, col_acc]]

NameError: name 'feature_imp' is not defined

## === cell 68
mdl = model[1]['model']
sc  = model[1]['scaler']

X_valid_sc = sc.transform(X_valid)#, columns=X_valid.columns)
y_pred_val = mdl.predict_proba(X_valid_sc)[:,1]
score      = metrics.roc_auc_score(y_valid, y_pred_val)

print('AUC em dados não viscto: {:2.5f}'.format(score))


## --- ERROR in cell 68, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2350566437.py in <cell line: 0>()
----> 1 mdl = model[1]['model']
      2 sc  = model[1]['scaler']
      3 
      4 X_valid_sc = sc.transform(X_valid)#, columns=X_valid.columns)
      5 y_pred_val = mdl.predict_proba(X_valid_sc)[:,1]

NameError: name 'model' is not defined

## === cell 69
explainer = shap.TreeExplainer(mdl)
shap_values = explainer.shap_values(X_valid_sc)

shap.summary_plot(shap_values, X_valid_sc, plot_type="bar")


## --- ERROR in cell 69, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/544171395.py in <cell line: 0>()
----> 1 explainer = shap.TreeExplainer(mdl)
      2 shap_values = explainer.shap_values(X_valid_sc)
      3 
      4 shap.summary_plot(shap_values, X_valid_sc, plot_type="bar")

NameError: name 'mdl' is not defined

## === cell 70
shap.summary_plot(shap_values, X_valid_sc, max_display=15)


## --- ERROR in cell 70, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3246398985.py in <cell line: 0>()
----> 1 shap.summary_plot(shap_values, X_valid_sc, max_display=15)

NameError: name 'shap_values' is not defined

## === cell 71
for erro in eval_metric:
    plt.figure(figsize=(15,10))

    for i in range(len(model)):
        results     = model[i]['model'].evals_result() # merror
        ntree_limit = model[i]['model'].best_ntree_limit

        plt.subplot(2,3,i+1)
        plt.plot(results["validation_0"][erro], label="Treinamento")
        plt.plot(results["validation_1"][erro], label="Validação")

        plt.axvline(ntree_limit, 
                    color="gray", 
                    label="N. de árvore ideal {}".format(ntree_limit))

        plt.ylabel(erro)
        plt.legend();

    plt.suptitle('Performance XGB - {}'.format(erro), y=1.05, fontsize=24);
    plt.tight_layout(h_pad=3.0);


## --- ERROR in cell 71, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/503238954.py in <cell line: 0>()
      2     plt.figure(figsize=(15,10))
      3 
----> 4     for i in range(len(model)):
      5         results     = model[i]['model'].evals_result() # merror
      6         ntree_limit = model[i]['model'].best_ntree_limit

NameError: name 'model' is not defined

## === cell 72
scaler_tr  = model[1]['scaler']
model_tr   = model[1]['model']
X_valid_sc = pd.DataFrame(scaler_tr.transform(X_valid), columns=X_valid.columns)

y_pred_prob = model_tr.predict_proba(X_valid_sc.values, ntree_limit=model_tr.best_ntree_limit)[:,1]
y_pred      = (y_pred_prob >.5).astype(int)
auc         = metrics.roc_auc_score(y_valid, y_pred_prob)

print('AUC: {:2.5f}'.format(auc))


## --- ERROR in cell 72, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3695781582.py in <cell line: 0>()
----> 1 scaler_tr  = model[1]['scaler']
      2 model_tr   = model[1]['model']
      3 X_valid_sc = pd.DataFrame(scaler_tr.transform(X_valid), columns=X_valid.columns)
      4 
      5 y_pred_prob = model_tr.predict_proba(X_valid_sc.values, ntree_limit=model_tr.best_ntree_limit)[:,1]

NameError: name 'model' is not defined

## === cell 73
def plot_cm(preds,true,ax=None):
    cm = confusion_matrix(preds.round(), true)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False, values_format = '.6g')
    plt.title('Confusion matrix \n')
    plt.grid(False)
    return disp

plot_cm(y_pred, y_valid);


## --- ERROR in cell 73, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/10394913.py in <cell line: 0>()
      7     return disp
      8 
----> 9 plot_cm(y_pred, y_valid);

NameError: name 'y_pred' is not defined

## === cell 74
threshold = .47

y_pred_threshold     = (y_pred_prob>threshold).astype(int)
fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred_threshold)

plot_roc_curve(fpr, tpr, label="XGB")
plt.show()

print('AUC     : {:2.5f} '.format(metrics.roc_auc_score(y_valid, y_pred_prob) ))
print('F1-score: {:2.5f}'.format(metrics.f1_score(y_valid, y_pred)))
print('F1-score: {:2.5f} threshold({:2.2f})'.format(metrics.f1_score(y_valid, y_pred_threshold), threshold))


## --- ERROR in cell 74, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3233221090.py in <cell line: 0>()
      2 
      3 y_pred_threshold     = (y_pred_prob>threshold).astype(int)
----> 4 fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred_threshold)
      5 
      6 plot_roc_curve(fpr, tpr, label="XGB")

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_ranking.py in roc_curve(y_true, y_score, pos_label, sample_weight, drop_intermediate)
    990     array([1.8 , 0.8 , 0.4 , 0.35, 0.1 ])
    991     """
--> 992     fps, tps, thresholds = _binary_clf_curve(
    993         y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    994     )

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_ranking.py in _binary_clf_curve(y_true, y_score, pos_label, sample_weight)
    749         raise ValueError("{0} format is not supported".format(y_type))
    750 
--> 751     check_consistent_length(y_true, y_score, sample_weight)
    752     y_true = column_or_1d(y_true)
    753     y_score = column_or_1d(y_score)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    395     uniques = np.unique(lengths)
    396     if len(uniques) > 1:
--> 397         raise ValueError(
    398             "Found input variables with inconsistent numbers of samples: %r"
    399             % [int(l) for l in lengths]

ValueError: Found input variables with inconsistent numbers of samples: [168000, 72000]
