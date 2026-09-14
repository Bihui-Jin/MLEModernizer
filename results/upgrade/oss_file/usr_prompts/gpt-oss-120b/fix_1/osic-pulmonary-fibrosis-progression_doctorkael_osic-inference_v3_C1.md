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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-6.847914207559203

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob

from tqdm import tqdm

from sklearn.model_selection import (
    cross_val_score, GroupKFold, GridSearchCV, 
    cross_val_predict, RandomizedSearchCV)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, OrdinalEncoder

plt.style.use("dark_background")
%matplotlib inline

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
main_dir = "../input/osic-pulmonary-fibrosis-progression"

!ls {main_dir}

## === cell 2
train_files = tf.io.gfile.glob(main_dir+"/train/*/*")
test_files = tf.io.gfile.glob(main_dir+"/test/*/*")
sample_sub = pd.read_csv(main_dir + "/sample_submission.csv")
train = pd.read_csv(main_dir + "/train.csv")
test = pd.read_csv(main_dir + "/test.csv")

print ("Number of train patients: {}\nNumber of test patients: {:4}"
       .format(train.Patient.nunique(), test.Patient.nunique()))

print ("\nTotal number of Train patient records: {}\nTotal number of Test patient records: {:6}"
       .format(len(train_files), len(test_files)))

train.shape, test.shape, sample_sub.shape

## === cell 3
def laplace_log_likelihood(y_true, y_pred, sigma=70):
    sigma_clipped = tf.maximum(sigma, 70)

    delta_clipped = tf.minimum(tf.abs(y_true - y_pred), 1000)
    
    delta_clipped = tf.cast(delta_clipped, dtype=tf.float32)
    sigma_clipped = tf.cast(sigma_clipped, dtype=tf.float32)
    
    score = - tf.sqrt(2.0) * delta_clipped / sigma_clipped - tf.math.log(tf.sqrt(2.0) * sigma_clipped)
    
    return tf.reduce_mean(score)

## === cell 4
def l1(s):
    def scorer_func(x, y, sigma=s):
        return laplace_log_likelihood(x, y, sigma=s).numpy()
    
    return make_scorer(scorer_func, greater_is_better=False)

## === cell 5
def base_shift(data, q=50):
    x = data.copy()

    temp = (x.groupby("Patient")
            .apply(lambda x: x.loc[int(
                np.percentile(x['Weeks'].index, q=q)
            ), ["Weeks", "FVC", "Percent"]]))

    temp.rename(
        {"Weeks": "Base_Week", 
         "FVC": "Base_FVC", 
         "Percent": "Base_Percent"}, 
        axis=1, inplace=True)

    x = x.merge(temp, on='Patient')

    x['Week_Offset'] = x['Weeks'] - x['Base_Week']
    
    return x

## === cell 6
def multi_baseweek_frame(data, display=True):
    '''Function to return multiple base week frames -> instead of creating one base week,
    creates several to help our model learn better and predict past and future data better.'''       
    
    op = data.merge(
        data[['Patient', 'Weeks', 'FVC', 'Percent']].rename(
            {"Weeks": "Base_Week", 
             "FVC": "Base_FVC", 
             "Percent": "Base_Percent"}, axis=1), 
        on='Patient')

    op['Week_Offset'] = op['Weeks'] - op['Base_Week']

    op = op[op['Week_Offset'] != 0]

    if display:
        print ("Number of Samples:{:5} -> {:5}\nNumber of Columns:{:5} -> {:5}".format(
            data.shape[0], op.shape[0], data.shape[1], op.shape[1]))
    
    return op.sort_values(by=['Patient', 'Base_Week']).reset_index(drop=True)

## === cell 7
def get_model_data(data, cat_cols, num_cols, to_drop, cat_method='1h', transform_stats=None,
                   age_bins=None, train=True, display_stats=True, math=None, factor=False):
    
    '''Our pipeline for this notebook. This portion is complex, could be written much more efficiently 
    using simple sklearn tools. I have this bad habit of reinventing the wheel from scratch.'''
    
    X = data.copy().reset_index(drop=True)
    
    
    if age_bins:    
        X['binned_age'] = pd.cut(X['Age'], bins=range(0, 101, 100//(age_bins-1))).cat.codes / age_bins
        to_drop = to_drop + ['Age']   
        
    if math:
        prod = np.ones(X.shape[0])
        if 'sin' in math:
            X['Sin_week'] = X.groupby("Patient")['Weeks'].apply(np.sin)
            prod = prod * X['Sin_week']
        if 'cos' in math:
            X['Cos_week'] = X.groupby("Patient")['Weeks'].apply(np.cos)
            prod = prod * X['Cos_week']
        if 'tan' in math:
            X['Tan_week'] = X.groupby("Patient")['Weeks'].apply(np.tan)
            prod = prod * X['Cos_week']
            
        if len(math) > 1:
            X['Math_Prod'] = prod
            
    if factor:
        X['factor'] = X['Base_FVC'] / X['Base_Percent']
        
    
    if cat_cols != []:
    
        if cat_method == 'ord': # ordinal encoding for tree based models
            global ordenc
            if train:
                from sklearn.preprocessing import OrdinalEncoder
                ordenc = OrdinalEncoder()
                X = X.merge(
                    pd.DataFrame(
                        ordenc.fit_transform(X[cat_cols]).astype(int),
                        columns=map(lambda x: x+"_ord", cat_cols)),
                    left_index=True, right_index=True)

            else:
                X = X.merge(
                    pd.DataFrame(
                        ordenc.transform(X[cat_cols]).astype(int),
                        columns=map(lambda x: x+"_ord", cat_cols)),
                    left_index=True, right_index=True)           

        elif cat_method == '1h': # one hot encoding
            global onehenc
            if train:
                onehenc = OneHotEncoder()
                X = X.merge(
                    pd.DataFrame(
                        onehenc.fit_transform(X[cat_cols]).todense(),
                        columns=[*np.concatenate(onehenc.categories_)]),
                    left_index=True, right_index=True)

            else:
                X = X.merge(
                    pd.DataFrame(
                        onehenc.transform(X[cat_cols]).todense(),
                        columns=[*np.concatenate(onehenc.categories_)]),
                    left_index=True, right_index=True)

        elif cat_method == 'poly': # polynomial feature encoding
            global cat_comb
            if train:
                cat_comb = np.array(
                    np.meshgrid(*[X[cat].unique() for cat in cat_cols])
                ).T.reshape(-1, len(cat_cols))

            for combination in cat_comb:
                name = "_".join(map(str, combination))
                X[name] = 1
                for i in range(len(cat_cols)):
                    X[name] = X[name] & (X[cat_cols[i]] == combination[i]).astype(int)

    to_drop = to_drop + cat_cols
                
    
    global stats
    if train:
        if transform_stats is None:
            stats = X.describe().T
        else:
            stats = transform_stats

    for col in num_cols:
        
        if (not train) and (col not in X.columns):
            continue
            
        X[col] = (X[col] - stats.loc[col, 'min']) / (stats.loc[col, 'max'] - stats.loc[col, 'min'])
        

    global x_cols
    if train:
        
        Y = X['FVC'].dropna()
        
        if display_stats:
        
            print (X.corr()['FVC'].abs().sort_values(ascending=False)[1:])
        
        X = X.drop(to_drop, axis=1)
        x_cols = X.columns
        
        return X, Y
    
    else:
        
        X = X.drop(to_drop, axis=1, errors='ignore')
        X = X[x_cols]
        
        return X

## === cell 8
def augment_train_cosine(data, n_similar=3, threshold=0.25, display_sample=True):
    
    '''
    - `n_similar` is number of patients we cluster at a time more the cluster, more eratic it gets.
    - `threshold` is used for as a measure to counter the influence of "outliers"
    '''
    
    from sklearn.metrics.pairwise import cosine_similarity

    temp = base_shift(data.copy(), q=0)

    
    temp['present_minus_past'] = temp.groupby("Patient")['FVC'].transform('diff').fillna(0)
    temp['Week_diff'] = temp.groupby("Patient")['Weeks'].transform('diff').fillna(0)
    temp['pms'] = (temp['present_minus_past'] / temp['Week_diff']).replace([np.inf, -np.inf]).fillna(0)
    temp['pms_min']  = temp.groupby("Patient")['pms'].transform('min')
    temp['pms_25']   = temp.groupby("Patient")['pms'].transform(lambda x: np.percentile(x, q=25))
    temp['pms_mean'] = temp.groupby("Patient")['pms'].transform('mean')
    temp['pms_75']   = temp.groupby("Patient")['pms'].transform(lambda x: np.percentile(x, q=75))
    temp['pms_max']  = temp.groupby("Patient")['pms'].transform('max')
    temp['pms_sum']  = temp.groupby("Patient")['pms'].transform('sum')
    
    temp['pmb_avg'] = (temp['Percent'] - temp['Base_Percent']).groupby(temp.Patient).transform("mean")
    temp['p_std']    = temp.groupby("Patient")['Percent'].transform('std')

    temp = temp.merge(
        pd.concat([
            temp.groupby("Patient").apply(
                lambda x: (x['Percent'].values[-1] - x['Percent'].values[0]) / 
                (x['Weeks'].values[-1] - x['Weeks'].values[0])).rename("Slope"),
            
            temp.groupby("Patient").apply(
                lambda x: x['Week_Offset'].iloc[np.argmax(x['pms'])]).rename("pmsw_max"),    
            temp.groupby("Patient").apply(
                lambda x: x['Week_Offset'].iloc[np.argmin(x['pms'])]).rename("pmsw_min")
            
        ], axis=1, ignore_index=False), on='Patient')

    temp = temp.groupby('Patient').head(1).reset_index(drop=True)

    temp['factor'] = temp['Base_FVC'] / temp['Base_Percent']

    train_cols = [
        'Patient', #'Sex', 'SmokingStatus'
        
        'pms_25', 'pms_mean', 'pms_75', 
        'pms_sum', 'pmsw_min', 'pmsw_max', 
        'pmb_avg', 'Slope', 'factor'
    ]
    
    cat_cols = np.intersect1d(['Sex', 'SmokingStatus'], train_cols)

    temp = temp[train_cols]
    temp = pd.get_dummies(temp, columns=cat_cols, drop_first=True, prefix='', prefix_sep='')

    n_similar += 1
    groups = (pd.DataFrame(
        np.argsort(
            cosine_similarity(temp.drop("Patient", 1), temp.drop('Patient', 1)))
        [:, -1:-n_similar-1:-1]))

    groups = groups[~pd.DataFrame(np.sort(groups.values, axis=1)).duplicated()]
    
    groups = groups.applymap(lambda x: temp.Patient.to_dict()[x]).apply(list, axis=1).to_dict()

    aug_data = []
    for group in tqdm(groups.values(), disable=not display_sample):

        temp = base_shift(train[train.Patient.isin(group)], q=0)

        temp['base_per_diff_from_mean'] = temp['Base_Percent'] - temp['Base_Percent'].unique().mean()
        temp['Percent_shifted'] = temp['Percent'] - temp['base_per_diff_from_mean']

        temp['base_week_diff_from_mean'] = temp['Base_Week'] - temp['Base_Week'].unique().mean()
        temp['Week_shifted'] = temp['Weeks'] - temp['base_week_diff_from_mean']

        temp = pd.merge_ordered(
            temp.drop(['Age', 'Sex', 'SmokingStatus', 'Base_Week', 'Week_Offset'], axis=1),

            (temp.groupby("Week_shifted")['Percent_shifted']
             .agg(['mean', 'count']).query(f'count > {n_similar * threshold}')
             .drop("count", 1)),

            on='Week_shifted', left_by='Patient'
        )

        aug_data.append(temp)

    temp = pd.concat(aug_data).reset_index(drop=True)

    temp[['base_per_diff_from_mean', 'base_week_diff_from_mean', 'Base_Percent', 'Base_FVC']] = (
        temp.groupby("Patient")[['base_per_diff_from_mean', 'base_week_diff_from_mean', 
                                 'Base_Percent', 'Base_FVC']].fillna(method='ffill'))

    temp['Week_aug'] = temp['Week_shifted'] + temp['base_week_diff_from_mean']

    temp['Percent_aug'] = np.where(
        temp['Percent'].isna(), 
        temp['mean'] + temp['base_per_diff_from_mean'],
        temp['Percent'])

    temp = temp[~temp['Percent_aug'].isna()]

    test_ids = temp['FVC'].isna()
    temp.loc[test_ids, 'FVC'] = LinearRegression().fit(
        temp.loc[~test_ids, ['Base_Percent', 'Percent_aug', 'Base_FVC']], 
        temp.loc[~test_ids, 'FVC']).predict(
        temp.loc[test_ids, ['Base_Percent', 'Percent_aug', 'Base_FVC']])

    temp = temp.groupby(["Patient", 'Week_aug']).mean().reset_index()

    temp = temp[['Patient', 'Week_aug', 'Percent_aug', 'FVC']]
    temp = temp.rename({'Week_aug': 'Weeks', 'Percent_aug': 'Percent'}, axis=1)
    
    temp['Weeks'] = temp['Weeks'].astype(int)
    
    temp = temp.merge(
        (data[['Patient', 'Sex', 'Age', 'SmokingStatus']]
         .groupby('Patient').head(1).reset_index(drop=True)), 
        
        on='Patient')
    
    if display_sample:
        
        print ("Data augmented by factor: {:.2f}x".format(1 + (
            temp.shape[0] - data.shape[0]) / data.shape[0]))

        f, ax = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
        for i, pat in enumerate(temp.Patient.unique()[:4]):

            ax[i][0].plot(*list(zip(*data[data.Patient == pat][['FVC', 'Weeks']].values))
                          [::-1], c='g', alpha=0.7)

            ax[i][1].plot(*list(zip(*data[data.Patient == pat][['Percent', 'Weeks']].values))
                          [::-1], c='g', alpha=0.7)

            ax[i][0].scatter(*list(zip(*temp[temp.Patient == pat][['FVC', 'Weeks']].values))
                          [::-1], c='r')

            ax[i][1].scatter(*list(zip(*temp[temp.Patient == pat][['Percent', 'Weeks']].values))
                          [::-1], c='r')
            

            ax[i][0].set(xlabel='Weeks', ylabel='FVC')
            ax[i][1].set(xlabel='Weeks', ylabel='Percent')
            f.suptitle("FVC & Percent Augmentation", y=.9)
    
    return temp

## === cell 9
def augment_train_naive(
    data, steps=5, method='index', noise=.25, val_split=0.25, 
    end_pts=[None, None], display_sample=True):
    
    '''
    end_pts -> start and end of augmentation, if None defaults to min/max for that patient
    '''
    
    temp = data[['Patient','Weeks', 'FVC', 'Percent']].merge(
        
        ((data.groupby("Patient")['Weeks']
         .apply(lambda x: pd.Series(
             np.union1d(np.arange(
                 end_pts[0] if end_pts[0] else x.min(), 
                 end_pts[1] if end_pts[1] else x.max(), 
                 step=steps), x))
               ).reset_index(level=0))),

        on=['Patient', 'Weeks'], how='right')

    temp.loc[:, ['FVC', 'Percent']] = (
        temp.groupby("Patient")[['FVC', 'Percent']]
        .apply(lambda x: (
            x.interpolate(method=method, limit_direction='both') + 
            
            (x.std().values * np.random.uniform(-noise, noise, [len(x), 1])))))

    temp = temp.merge(
        data.groupby("Patient")[['Patient', 'Age', 'Sex', 'SmokingStatus']].head(1),
        on='Patient')
    
    if display_sample:
        f, ax = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))

        for i, pat in enumerate(temp.Patient.unique()[:4]):

            ax[i][0].plot(*list(zip(*data[data.Patient == pat][['FVC', 'Weeks']].values))
                [::-1], c='g', alpha=0.7)
            
            ax[i][1].plot(*list(zip(*data[data.Patient == pat][['Percent', 'Weeks']].values))
                          [::-1], c='g', alpha=0.7)
            
            ax[i][0].scatter(*list(zip(*temp[temp.Patient == pat][['FVC', 'Weeks']].values))[::-1], c='r')
            
            ax[i][1].scatter(*list(zip(*temp[temp.Patient == pat][['Percent', 'Weeks']].values))[::-1], c='r')
            
            ax[i][0].set(xlabel='Weeks', ylabel='FVC')
            ax[i][1].set(xlabel='Weeks', ylabel='Percent')
            f.suptitle("FVC & Percent Augmentation", y=.9)
            
        print ("Data augmented by factor: {:.2f}x".format(1 + (
            temp.shape[0] - data.shape[0]) / data.shape[0]))
    
    return temp

## === cell 10
sub = sample_sub.Patient_Week.str.extract("(ID\w+)_(\-?\d+)").rename({0: "Patient", 1: "Weeks"}, axis=1)
sub['Weeks'] = sub['Weeks'].astype(int)
sub = pd.merge(sub, test[['Patient', 'Sex', 'SmokingStatus']], on='Patient')
sub["Patient_Week"] = sub.Patient + "_" + sub.Weeks.astype(str)
sub.head()

## === cell 11
class GBR(RegressorMixin, BaseEstimator):
    def __init__(self, alpha=.75, **params):
        self.alpha = alpha
        self.umodel = self._create_model(loss='quantile', q=self.alpha, **params)
        self.mmodel = self._create_model(loss='lad', **params)
        self.lmodel = self._create_model(loss='quantile', q=1-self.alpha, **params)
              
    def _create_model(self, loss, q=.75, **params):
        model = GradientBoostingRegressor(
            init=LinearRegression(),
            criterion='friedman_mse',
            n_estimators=50, max_depth=2, 
            loss=loss, alpha=q, **params)
        
        return model
        
    def fit(self, x, y):
        
        self.umodel.fit(x, y)
        self.mmodel.fit(x, y)
        self.lmodel.fit(x, y)
        return self
    
    def predict(self, X):
        
        return self.mmodel.predict(X)
    
    def predict_forecast(self, X, return_bounds=False):
        
        preds = self.mmodel.predict(X)
        upper = self.umodel.predict(X)
        lower = self.lmodel.predict(X)
        
        if return_bounds:
            return preds, upper, lower
        else:
            return preds, (upper - lower)

## === cell 12
stats = multi_baseweek_frame(pd.concat([train, test]), display=False).describe().T

## === cell 13
op = multi_baseweek_frame(
    augment_train_cosine(train, display_sample=False, n_similar=5, threshold=.5)
)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3874514907.py in <cell line: 0>()
      1 op = multi_baseweek_frame(
----> 2     augment_train_cosine(train, display_sample=False, n_similar=5, threshold=.5)
      3 )

/tmp/ipykernel_11/2145335653.py in augment_train_cosine(data, n_similar, threshold, display_sample)
     68         np.argsort(
     69             # cosine similarity to get their similarity scores
---> 70             cosine_similarity(temp.drop("Patient", 1), temp.drop('Patient', 1)))
     71         [:, -1:-n_similar-1:-1]))
     72 

TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 were given

## === cell 14
cat_cols = ['Sex', 'SmokingStatus']
to_drop = ['FVC', 'Percent', 'Weeks', 'factor', 'Base_Percent']
num_cols = ['Weeks', 'Week_Offset', 'Base_Week', 'Age', 'Base_FVC', 'Percent', 'Base_Percent']
cat_method = 'ord'
math = []
age_bins = 5

folds = 7
total_patients = train.Patient.unique()
np.random.shuffle(total_patients)
val_len = len(total_patients) // folds

temp = pd.DataFrame()
preds = pd.DataFrame()

X, Y = get_model_data(
    op, factor=True, num_cols=num_cols, cat_cols=cat_cols, 
    age_bins=age_bins, to_drop=to_drop, display_stats=False, 
    cat_method=cat_method, transform_stats=stats, math=math)

X_VAL = base_shift(train, q=0)
Y_VAL = X_VAL['FVC'].dropna()

X_VAL['Percent'] = X_VAL['Base_Percent']

X_VAL = get_model_data(
    X_VAL, num_cols=num_cols, cat_cols=cat_cols, to_drop=to_drop, factor=True,
    cat_method=cat_method, train=False, age_bins=age_bins, math=math)

x_test = sub[['Patient', 'Weeks']].merge(
    test.rename({"Weeks": "Base_Week", 
                 "FVC": "Base_FVC", 
                 "Percent": "Base_Percent"}, axis=1), 
    on='Patient')

x_test['Week_Offset'] = x_test['Weeks'] - x_test['Base_Week']

x_test['Percent'] = x_test['Base_Percent']

x_test = get_model_data(
    x_test, cat_cols=cat_cols, num_cols=num_cols, to_drop=to_drop, math=math,
    factor=True, train=False, cat_method=cat_method, 
    age_bins=age_bins).drop("Patient", 1)

for i in range(folds):
   
    val_patients = total_patients[(i)*val_len:(i+1)*val_len]
    train_patients = np.setdiff1d(total_patients, val_patients)
    
    assert len(np.intersect1d(val_patients, train_patients)) == 0
    
    x, y, = (X[X.Patient.isin(train_patients)].drop("Patient", 1), Y[X.Patient.isin(train_patients)])
    x_val, y_val = (X_VAL[X_VAL.Patient.isin(val_patients)].drop("Patient", 1), 
                    Y_VAL[X_VAL.Patient.isin(val_patients)])
        
    model = GBR(alpha=0.75)

    model.fit(x, y)
    y_middle, y_upper, y_lower = model.predict_forecast(x_val, return_bounds=True)
    y_middle_pred, y_test_conf = model.predict_forecast(x_test)
    
    print ("For Fold #{} Val Score: {:.2f} @ 70 Confidence | {:.2f} @ Pred Confidence".format(
        i+1, - laplace_log_likelihood(y_middle, y_val), 
        - laplace_log_likelihood(y_middle, y_val, y_upper - y_lower)))
    
    temp = temp.append(pd.DataFrame(
        data=np.stack([y_upper, y_lower, y_middle, y_val], axis=1),
        columns=['upper', 'lower', 'pred', 'actual']
    ))
    
    preds = preds.append(pd.DataFrame(
        data=np.stack([y_middle_pred, y_test_conf], axis=1) / folds,
        columns=['pred', 'Confidence']
    ))
    
preds = preds.groupby(preds.index).sum()
temp['Confidence'] = temp['upper'] - temp['lower']

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3197873069.py in <cell line: 0>()
     19 # creating data suitable for model fitting and predictions
     20 X, Y = get_model_data(
---> 21     op, factor=True, num_cols=num_cols, cat_cols=cat_cols,
     22     age_bins=age_bins, to_drop=to_drop, display_stats=False,
     23     cat_method=cat_method, transform_stats=stats, math=math)

NameError: name 'op' is not defined

## === cell 15
pat_scores = (
    temp.reset_index(drop=True)
    .groupby(train.Patient)
     .apply(lambda x: -laplace_log_likelihood(x['actual'], x['pred'], x['Confidence']).numpy())
).rename('scores').reset_index().sort_values("scores", ascending=False)

pat_scores = pat_scores.head(8)
print ("Worst Patient-Mean-Score: {:.2f}".format(pat_scores.scores.mean()))
pat_scores = pat_scores.Patient.values

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2748953141.py in <cell line: 0>()
      3     .groupby(train.Patient)
      4      .apply(lambda x: -laplace_log_likelihood(x['actual'], x['pred'], x['Confidence']).numpy())
----> 5 ).rename('scores').reset_index().sort_values("scores", ascending=False)
      6 
      7 pat_scores = pat_scores.head(8)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in rename(self, mapper, index, columns, axis, copy, inplace, level, errors)
   5765         4  3  6
   5766         """
-> 5767         return super()._rename(
   5768             mapper=mapper,
   5769             index=index,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _rename(self, mapper, index, columns, axis, copy, inplace, level, errors)
   1120                     indexer = ax.get_level_values(level).get_indexer_for(replacements)
   1121                 else:
-> 1122                     indexer = ax.get_indexer_for(replacements)
   1123 
   1124                 if errors == "raise" and len(indexer[indexer == -1]):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_indexer_for(self, target)
   6180         """
   6181         if self._index_as_unique:
-> 6182             return self.get_indexer(target)
   6183         indexer, _ = self.get_indexer_non_unique(target)
   6184         return indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_indexer(self, target, method, limit, tolerance)
   3878         method = clean_reindex_fill_method(method)
   3879         orig_target = target
-> 3880         target = self._maybe_cast_listlike_indexer(target)
   3881 
   3882         self._check_indexing_method(method, limit, tolerance)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _maybe_cast_listlike_indexer(self, target)
   6681         Analogue to maybe_cast_indexer for get_indexer instead of get_loc.
   6682         """
-> 6683         return ensure_index(target)
   6684 
   6685     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in ensure_index(index_like, copy)
   7647             return Index(index_like, copy=copy, tupleize_cols=False)
   7648     else:
-> 7649         return Index(index_like, copy=copy)
   7650 
   7651 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in __new__(cls, data, dtype, copy, name, tupleize_cols)
    524 
    525         elif is_scalar(data):
--> 526             raise cls._raise_scalar_data_error(data)
    527         elif hasattr(data, "__array__"):
    528             return cls(np.asarray(data), dtype=dtype, copy=copy, name=name)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_scalar_data_error(cls, data)
   5287         # We return the TypeError so that we can raise it from the constructor
   5288         #  in order to keep mypy happy
-> 5289         raise TypeError(
   5290             f"{cls.__name__}(...) must be called with a collection of some "
   5291             f"kind, {repr(data) if not isinstance(data, np.generic) else str(data)} "

TypeError: Index(...) must be called with a collection of some kind, 'scores' was passed

## === cell 16
print ("\n|================== Summary ==================|\n\
Score on Total Dataset: {:.3f} @   70 Confidence\n\
Score on Total Dataset: {:.3f} @  225 Confidence\n\
Score on Total Dataset: {:.3f} @ Pred Confidence".format(
    -laplace_log_likelihood(temp['actual'], temp['pred'], 70),
    -laplace_log_likelihood(temp['actual'], temp['pred'], 225),
    -laplace_log_likelihood(temp['actual'], temp['pred'], temp['Confidence'])
))

f, ax = plt.subplots(figsize=(40, 40), nrows=4, ncols=2)
ax = ax.ravel()
for i, pat in enumerate(pat_scores):
    (temp.reset_index(drop=True).loc[train.Patient == pat]
     .drop(["Confidence"], 1).plot(ax=ax[i], legend=False))
f.suptitle("Model's Worst Predictions", size=30)
f.tight_layout(rect=[0, 0.03, 1, 0.95]);

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3637742149.py in <cell line: 0>()
      3 Score on Total Dataset: {:.3f} @  225 Confidence\n\
      4 Score on Total Dataset: {:.3f} @ Pred Confidence".format(
----> 5     -laplace_log_likelihood(temp['actual'], temp['pred'], 70),
      6     -laplace_log_likelihood(temp['actual'], temp['pred'], 225),
      7     -laplace_log_likelihood(temp['actual'], temp['pred'], temp['Confidence'])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'actual'

## === cell 17
sub['FVC'] = preds['pred']
sub['Confidence'] = preds['Confidence']

for i in range(len(test)):
    sub.loc[sub['Patient_Week']==test.Patient[i]+'_'+str(test.Weeks[i]), 'FVC'] = test.FVC[i]
    sub.loc[sub['Patient_Week']==test.Patient[i]+'_'+str(test.Weeks[i]), 'Confidence'] = 70

sub[['Patient_Week', 'FVC', 'Confidence']].to_csv("quant_submission.csv", index=False)
sub.head()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3471886774.py in <cell line: 0>()
----> 1 sub['FVC'] = preds['pred']
      2 sub['Confidence'] = preds['Confidence']
      3 
      4 # final touches before submission
      5 for i in range(len(test)):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'pred'

## === cell 18
cat_cols = ['Sex', 'SmokingStatus']
num_cols = ['Weeks', 'Week_Offset', 'Base_Week', 'Age', 'Base_FVC', 'Percent', 'Base_Percent']
to_drop = ["FVC", 'Percent', 'Weeks', 'Base_Week', 'Age', 'Base_Percent', 'factor']
math = []

multi_method = {}
multi_data = {}
folds = 7

methods = ['cubic', 'quadratic', 'cubicspline', 
           'pchip', 'akima', 'nearest', 'zero', 
           'slinear', 'linear', 'SIMILARITY_AUG']

total_patients = train.Patient.unique()
val_len = len(total_patients) // folds

for method in methods:
    
    if method == 'SIMILARITY_AUG':
        temp = augment_train_cosine(train, n_similar=3, threshold=0.25, display_sample=False)
    else:
        temp = augment_train_naive(data=train, method=method, steps=15, noise=0., display_sample=False)
        
    temp = multi_baseweek_frame(temp, display=False)
    multi_data[method] = temp
    
    X, Y = get_model_data(temp, num_cols=num_cols, cat_cols=cat_cols, to_drop=to_drop, 
        display_stats=False, cat_method='1h', math=math, factor=True)
        
    X_VAL = base_shift(train, q=0)
    Y_VAL = X_VAL['FVC'].dropna()
    X_VAL['Percent'] = X_VAL['Base_Percent']
    X_VAL = get_model_data(
        X_VAL, num_cols=num_cols, cat_cols=cat_cols, factor=True,
        to_drop=to_drop, train=False, cat_method='1h', math=math)
    
    np.random.shuffle(total_patients)
    scores = {}
    
    for i in range(folds):
   
        val_patients = total_patients[(i)*val_len:(i+1)*val_len]
        train_patients = np.setdiff1d(total_patients, val_patients)

        x, y = X[X.Patient.isin(train_patients)], Y[X.Patient.isin(train_patients)]
        x_val, y_val = X_VAL[X_VAL.Patient.isin(val_patients)], Y_VAL[X_VAL.Patient.isin(val_patients)]
        
        assert len(np.intersect1d(x_val.Patient.unique(), x.Patient.unique())) == 0
        
        scores['Train'] = scores.get('Train', []) + [cross_val_score(
            GBR(), x.drop("Patient", 1), y, 
            scoring=l1(70), cv=GroupKFold(5), groups=x.Patient).mean()]

        lr = GBR().fit(x.drop("Patient", 1), y) 
        temp = lr.predict_forecast(x_val.drop("Patient", 1))
        temp = pd.DataFrame(np.stack(temp, 1), columns=['pred', 'conf'])
        temp['actual'] = y_val.reset_index(drop=True)
        
        scores['Val'] = scores.get('Val', []) + [-laplace_log_likelihood(
            temp['actual'], temp['pred'], 70
        ).numpy()]
        
        scores['ValC'] = scores.get('ValC', []) + [-laplace_log_likelihood(
            temp['actual'], temp['pred'], temp['conf']
        ).numpy()]
        
        scores['ValW'] = scores.get('ValW', []) + [temp.apply(lambda x: -laplace_log_likelihood(
            x['actual'], x['pred'], x['conf']).numpy(), 1).nlargest(25).mean()]
        
    print ("Method: {}\nTrain score: {:5.2f} @ {:.2f} Variance \
    \nVal Score: {:7.2f} @ {:.2f} Variance\
    \nValC Score: {:6.2f} @ {:.2f} Variance\
    \nWorst Score: {:5.2f} @ {:.2f} Variance\n{}\n".format(
        method.upper(), np.mean(scores['Train']), np.std(scores['Train']), 
        np.mean(scores['Val']), np.std(scores['Val']), np.mean(scores['ValC']), 
        np.std(scores['ValC']), np.mean(scores['ValW']), np.std(scores['ValW']), "=" * 35
    ))
    
    multi_method[method] = scores

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/689795674.py in <cell line: 0>()
     22         temp = augment_train_cosine(train, n_similar=3, threshold=0.25, display_sample=False)
     23     else:
---> 24         temp = augment_train_naive(data=train, method=method, steps=15, noise=0., display_sample=False)
     25 
     26     # create multi base week data

/tmp/ipykernel_11/3794067867.py in augment_train_naive(data, steps, method, noise, val_split, end_pts, display_sample)
     19         on=['Patient', 'Weeks'], how='right')
     20 
---> 21     temp.loc[:, ['FVC', 'Percent']] = (
     22         temp.groupby("Patient")[['FVC', 'Percent']]
     23         .apply(lambda x: (

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __setitem__(self, key, value)
    909 
    910         iloc = self if self.name == "iloc" else self.obj.iloc
--> 911         iloc._setitem_with_indexer(indexer, value, self.name)
    912 
    913     def _validate_key(self, key, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _setitem_with_indexer(self, indexer, value, name)
   1940         if take_split_path:
   1941             # We have to operate column-wise
-> 1942             self._setitem_with_indexer_split_path(indexer, value, name)
   1943         else:
   1944             self._setitem_single_block(indexer, value, name)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _setitem_with_indexer_split_path(self, indexer, value, name)
   1975         if is_list_like_indexer(value) and getattr(value, "ndim", 1) > 0:
   1976             if isinstance(value, ABCDataFrame):
-> 1977                 self._setitem_with_indexer_frame_value(indexer, value, name)
   1978 
   1979             elif np.ndim(value) == 2:

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _setitem_with_indexer_frame_value(self, indexer, value, name)
   2098                 if item in value:
   2099                     sub_indexer[1] = item
-> 2100                     val = self._align_series(
   2101                         tuple(sub_indexer),
   2102                         value[item],

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _align_series(self, indexer, ser, multiindex_indexer, using_cow)
   2425                         return ser._values.copy()
   2426 
-> 2427                     return ser.reindex(new_ix)._values
   2428 
   2429                 # 2 dims

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in reindex(self, index, axis, method, copy, level, fill_value, limit, tolerance)
   5151         tolerance=None,
   5152     ) -> Series:
-> 5153         return super().reindex(
   5154             index=index,
   5155             method=method,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in reindex(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)
   5608 
   5609         # perform the reindex on the axes
-> 5610         return self._reindex_axes(
   5611             axes, level, limit, tolerance, method, fill_value, copy
   5612         ).__finalize__(self, method="reindex")

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _reindex_axes(self, axes, level, limit, tolerance, method, fill_value, copy)
   5631 
   5632             ax = self._get_axis(a)
-> 5633             new_index, indexer = ax.reindex(
   5634                 labels, level=level, limit=limit, tolerance=tolerance, method=method
   5635             )

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in reindex(self, target, method, level, limit, tolerance)
   4431                     indexer, _ = self.get_indexer_non_unique(target)
   4432 
-> 4433         target = self._wrap_reindex_result(target, indexer, preserve_names)
   4434         return target, indexer
   4435 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in _wrap_reindex_result(self, target, indexer, preserve_names)
   2715             else:
   2716                 try:
-> 2717                     target = MultiIndex.from_tuples(target)
   2718                 except TypeError:
   2719                     # not all tuples, see test_constructor_dict_multiindex_reindex_flat

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in new_meth(self_or_cls, *args, **kwargs)
    220             kwargs["names"] = kwargs.pop("name")
    221 
--> 222         return meth(self_or_cls, *args, **kwargs)
    223 
    224     return cast(F, new_meth)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/multi.py in from_tuples(cls, tuples, sortorder, names)
    615                 tuples = np.asarray(tuples._values)
    616 
--> 617             arrays = list(lib.tuples_to_object_array(tuples).T)
    618         elif isinstance(tuples, list):
    619             arrays = list(lib.to_object_array_tuples(tuples).T)

lib.pyx in pandas._libs.lib.tuples_to_object_array()

ValueError: Buffer dtype mismatch, expected 'Python object' but got 'long'

## === cell 19
val_scores = [np.mean(multi_method[i]['ValW']) for i in methods]
cut_off = np.percentile(val_scores, 50)
        
print("At {:.1f} cutoff, the Worst Score would be {:.3f} @ Pred Conf".format(
    cut_off, np.mean(list(filter(lambda x: x < cut_off, val_scores)))))

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3746844367.py in <cell line: 0>()
----> 1 val_scores = [np.mean(multi_method[i]['ValW']) for i in methods]
      2 cut_off = np.percentile(val_scores, 50)
      3 
      4 print("At {:.1f} cutoff, the Worst Score would be {:.3f} @ Pred Conf".format(
      5     cut_off, np.mean(list(filter(lambda x: x < cut_off, val_scores)))))

/tmp/ipykernel_11/3746844367.py in <listcomp>(.0)
----> 1 val_scores = [np.mean(multi_method[i]['ValW']) for i in methods]
      2 cut_off = np.percentile(val_scores, 50)
      3 
      4 print("At {:.1f} cutoff, the Worst Score would be {:.3f} @ Pred Conf".format(
      5     cut_off, np.mean(list(filter(lambda x: x < cut_off, val_scores)))))

KeyError: 'cubic'

## === cell 20
x_test = sub[['Patient', 'Weeks']].merge(
    test.rename({"Weeks": "Base_Week", 
                 "FVC": "Base_FVC", 
                 "Percent": "Base_Percent"}, axis=1), 
    on='Patient')

x_test['Week_Offset'] = x_test['Weeks'] - x_test['Base_Week']

x_test['Percent'] = x_test['Base_Percent']

preds = {i: None for i in methods}

for method in methods:
    
    if np.mean(multi_method[method]['ValW']) >= cut_off:
        preds.pop(method)
        continue
    
    x, y = get_model_data(
        multi_data[method], num_cols=num_cols, cat_cols=cat_cols, 
        to_drop=to_drop, train=True, cat_method='1h', 
        display_stats=False, factor=True,
    )

    lr = GBR().fit(x.drop("Patient", 1), y) 

    preds[method] = lr.predict_forecast(get_model_data(
        x_test, cat_cols=cat_cols, num_cols=num_cols, to_drop=to_drop, train=False, factor=True,
    ).drop("Patient", 1), return_bounds=True)

preds.keys()

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2019302799.py in <cell line: 0>()
     18 for method in methods:
     19 
---> 20     if np.mean(multi_method[method]['ValW']) >= cut_off:
     21         preds.pop(method)
     22         continue

KeyError: 'cubic'

## === cell 21
temp = {}
for i in multi_method:
    if i in preds.keys():
        temp['Val'] = temp.get('Val', []) + [np.mean(multi_method[i]['Val'])]
        temp['ValC'] = temp.get('ValC', []) + [np.mean(multi_method[i]['ValC'])]
        
print ("At same cutoff:\n\nBest val score: {:6.3f} @   70 Conf\n\
Best Val score: {:6.3f} @ Pred Conf".format(np.mean(temp['Val']), np.mean(temp['ValC'])))

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/308672987.py in <cell line: 0>()
      6 
      7 print ("At same cutoff:\n\nBest val score: {:6.3f} @   70 Conf\n\
----> 8 Best Val score: {:6.3f} @ Pred Conf".format(np.mean(temp['Val']), np.mean(temp['ValC'])))

KeyError: 'Val'

## === cell 22
mean = np.mean(np.stack(pd.DataFrame(preds).iloc[0].values), 0)
upper = np.max(np.stack(pd.DataFrame(preds).iloc[1].values), 0)
lower = np.min(np.stack(pd.DataFrame(preds).iloc[2].values), 0)

sub['FVC'] = mean
sub['Confidence'] = upper - lower

sub[['Patient_Week', 'FVC', 'Confidence']].to_csv("submission.csv", index=False)

sub.head()

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/122107962.py in <cell line: 0>()
----> 1 mean = np.mean(np.stack(pd.DataFrame(preds).iloc[0].values), 0)
      2 upper = np.max(np.stack(pd.DataFrame(preds).iloc[1].values), 0)
      3 lower = np.min(np.stack(pd.DataFrame(preds).iloc[2].values), 0)
      4 
      5 sub['FVC'] = mean

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    665 
    666     if not indexes and not raw_lengths:
--> 667         raise ValueError("If using all scalar values, you must pass an index")
    668 
    669     if have_series:

ValueError: If using all scalar values, you must pass an index
