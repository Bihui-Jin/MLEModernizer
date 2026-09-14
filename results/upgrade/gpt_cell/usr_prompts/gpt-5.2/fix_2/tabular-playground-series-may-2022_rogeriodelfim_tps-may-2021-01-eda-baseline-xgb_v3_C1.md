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

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
lines = int(len(feature_float) / 2)
fig, ax = plt.subplots(lines, 2, figsize=(20, 20))

for i, feature in enumerate(feature_float):
    plt.subplot(lines, 2, i + 1)

    x_train = df2_train[feature].astype(np.float32, copy=False)
    x_test = df2_test[feature].astype(np.float32, copy=False)

    sns.histplot(
        data=df2_train, x=x_train, color="blue", alpha=0.5, label="train", bins=1000
    )
    sns.histplot(
        data=df2_test, x=x_test, color="teal", alpha=0.5, label="test", bins=1000
    )
    plt.xlabel(feature, fontsize=12)
    plt.legend()

plt.suptitle("DistPlot: train & test data", fontsize=20)


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3659953496.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mplt[0m[0;34m.[0m[0mfigure[0m[0;34m([0m[0mfigsize[0m[0;34m=[0m[0;34m([0m[0;36m20[0m[0;34m,[0m[0;36m15[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mgraf_feature_corr[0m[0;34m([0m[0mdf[0m[0;34m=[0m[0mdf2_train[0m[0;34m.[0m[0mcopy[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0;34m'id'[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m,[0m [0mannot_[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mthreshold[0m[0;34m=[0m[0;36m.5[0m[0;34m,[0m [0mprint_var[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/1695127873.py[0m in [0;36mgraf_feature_corr[0;34m(df, annot_, threshold, print_var)[0m
[1;32m      1[0m [0;32mdef[0m [0mgraf_feature_corr[0m[0;34m([0m[0mdf[0m[0;34m,[0m [0mannot_[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mthreshold[0m[0;34m=[0m[0;36m.8[0m[0;34m,[0m [0mprint_var[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m     [0mdf[0m [0;34m=[0m [0mdf[0m[0;34m.[0m[0mcorr[0m[0;34m([0m[0mmethod[0m [0;34m=[0m[0;34m'pearson'[0m[0;34m)[0m[0;34m.[0m[0mround[0m[0;34m([0m[0;36m5[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m     [0;31m# Máscara para ocultar a parte superior direita do gráfico, pois é uma duplicata[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mcorr[0;34m(self, method, min_periods, numeric_only)[0m
[1;32m  11047[0m         [0mcols[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mcolumns[0m[0;34m[0m[0;34m[0m[0m
[1;32m  11048[0m         [0midx[0m [0;34m=[0m [0mcols[0m[0;34m.[0m[0mcopy[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m> 11049[0;31m         [0mmat[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mto_numpy[0m[0;34m([0m[0mdtype[0m[0;34m=[0m[0mfloat[0m[0;34m,[0m [0mna_value[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mnan[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m  11050[0m [0;34m[0m[0m
[1;32m  11051[0m         [0;32mif[0m [0mmethod[0m [0;34m==[0m [0;34m"pearson"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mto_numpy[0;34m(self, dtype, copy, na_value)[0m
[1;32m   1991[0m         [0;32mif[0m [0mdtype[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1992[0m             [0mdtype[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mdtype[0m[0;34m([0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1993[0;31m         [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_mgr[0m[0;34m.[0m[0mas_array[0m[0;34m([0m[0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0mcopy[0m[0;34m,[0m [0mna_value[0m[0;34m=[0m[0mna_value[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1994[0m         [0;32mif[0m [0mresult[0m[0;34m.[0m[0mdtype[0m [0;32mis[0m [0;32mnot[0m [0mdtype[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1995[0m             [0mresult[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mresult[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36mas_array[0;34m(self, dtype, copy, na_value)[0m
[1;32m   1692[0m                 [0marr[0m[0;34m.[0m[0mflags[0m[0;34m.[0m[0mwriteable[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1693[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1694[0;31m             [0marr[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_interleave[0m[0;34m([0m[0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m,[0m [0mna_value[0m[0;34m=[0m[0mna_value[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1695[0m             [0;31m# The underlying data was copied within _interleave, so no need[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1696[0m             [0;31m# to further copy if copy=True or setting na_value[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36m_interleave[0;34m(self, dtype, na_value)[0m
[1;32m   1751[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1752[0m                 [0marr[0m [0;34m=[0m [0mblk[0m[0;34m.[0m[0mget_values[0m[0;34m([0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1753[0;31m             [0mresult[0m[0;34m[[0m[0mrl[0m[0;34m.[0m[0mindexer[0m[0;34m][0m [0;34m=[0m [0marr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1754[0m             [0mitemmask[0m[0;34m[[0m[0mrl[0m[0;34m.[0m[0mindexer[0m[0;34m][0m [0;34m=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1755[0m [0;34m[0m[0m

[0;31mValueError[0m: could not convert string to float: 'ACBADABECB'

## === cell 55
plt.subplots(figsize=(20, 15))

for i, col in enumerate(feature_float):    
    plt.subplot(int(len(feature_float)/3 +1),3,i+1)
    sns.kdeplot(data=df2_train, x=col, hue=target, legend=True, shade=True, multiple='stack');
