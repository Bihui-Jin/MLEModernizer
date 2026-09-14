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

category_encoders==2.7.0
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
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


## === cell 1
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


## === cell 2
df_train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
df_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


## === cell 3
print(f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}')
print(f'Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
print(f'Test Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f'Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')


## === cell 4
training_sample_counts = df_train.rename(columns={'Weeks': 'Samples'}).groupby('Patient').agg('count')['Samples'].value_counts()
print(f'Training Set FVC Measurements Per Patient \n{("-") * 41}\n{training_sample_counts}')


## === cell 5
df_test.head(2)


## === cell 6
df_submission = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv' )
df_submission.head()


## === cell 7
print(f'FVC Statistical Summary\n{"-" * 23}')

print(f'Mean: {df_train["FVC"].mean():.6}  -  Median: {df_train["FVC"].median():.6}  -  Std: {df_train["FVC"].std():.6}')
print(f'Min: {df_train["FVC"].min()}  -  25%: {df_train["FVC"].quantile(0.25)}  -  50%: {df_train["FVC"].quantile(0.5)}  -  75%: {df_train["FVC"].quantile(0.75)}  -  Max: {df_train["FVC"].max()}')
print(f'Skew: {df_train["FVC"].skew():.6}  -  Kurtosis: {df_train["FVC"].kurtosis():.6}')
missing_values_count = df_train[df_train["FVC"].isnull()].shape[0]
training_samples_count = df_train.shape[0]
print(f'Missing Values: {missing_values_count}/{training_samples_count} ({missing_values_count * 100 / training_samples_count:.4}%)')

fig, axes = plt.subplots(ncols=2, figsize=(18, 6), dpi=150)

sns.distplot(df_train['FVC'], label='FVC', ax=axes[0])
stats.probplot(df_train['FVC'], plot=axes[1])

for i in range(2):
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    
axes[0].set_title(f'FVC Distribution in Training Set', size=15, pad=15)
axes[1].set_title(f'FVC Probability Plot', size=15, pad=15)

plt.show()


## === cell 8
def plot_fvc(df, patient):
        
    df[['Weeks', 'FVC']].set_index('Weeks').plot(figsize=(30, 6), label='_nolegend_')
    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(f'Patient: {patient} - {df["Age"].tolist()[0]} - {df["Sex"].tolist()[0]} - {df["SmokingStatus"].tolist()[0]} ({len(df)} Measurements in {(df["Weeks"].max() - df["Weeks"].min())} Weeks Period)', size=25, pad=25)
    plt.legend().set_visible(False)
    plt.show()

for patient, df in list(df_train.groupby('Patient')):
    
    df['FVC_diff-1'] = np.abs(df['FVC'].diff(-1))
    
    print(f'Patient: {patient} FVC Statistical Summary\n{"-" * 58}')
    print(f'Mean: {df["FVC"].mean():.6}  -  Median: {df["FVC"].median():.6}  -  Std: {df["FVC"].std():.6}')
    print(f'Min: {df["FVC"].min()} -  Max: {df["FVC"].max()}')
    print(f'Skew: {df["FVC"].skew():.6}  -  Kurtosis: {df["FVC"].kurtosis():.6}')
    print(f'Change Mean: {df["FVC_diff-1"].mean():.6}  - Change Median: {df["FVC_diff-1"].median():.6}  - Change Std: {df["FVC_diff-1"].std():.6}')
    print(f'Change Min: {df["FVC_diff-1"].min()} -  Change Max: {df["FVC_diff-1"].max()}')
    print(f'Change Skew: {df["FVC_diff-1"].skew():.6} -  Change Kurtosis: {df["FVC_diff-1"].kurtosis():.6}')
    
    plot_fvc(df, patient)


## === cell 10
g = sns.pairplot(df_train[['FVC', 'Weeks', 'Percent', 'Age']], aspect=1.4, height=5, diag_kind='kde', kind='reg')

g.axes[3, 0].set_xlabel('FVC', fontsize=20)
g.axes[3, 1].set_xlabel('Weeks', fontsize=20)
g.axes[3, 2].set_xlabel('Percent', fontsize=20)
g.axes[3, 3].set_xlabel('Age', fontsize=20)
g.axes[0, 0].set_ylabel('FVC', fontsize=20)
g.axes[1, 0].set_ylabel('Weeks', fontsize=20)
g.axes[2, 0].set_ylabel('Percent', fontsize=20)
g.axes[3, 0].set_ylabel('Age', fontsize=20)

g.axes[3, 0].tick_params(axis='x', labelsize=15)
g.axes[3, 1].tick_params(axis='x', labelsize=15)
g.axes[3, 2].tick_params(axis='x', labelsize=15)
g.axes[3, 3].tick_params(axis='x', labelsize=15)
g.axes[0, 0].tick_params(axis='y', labelsize=15)
g.axes[1, 0].tick_params(axis='y', labelsize=15)
g.axes[2, 0].tick_params(axis='y', labelsize=15)
g.axes[3, 0].tick_params(axis='y', labelsize=15)

g.fig.suptitle('Tabular Data Feature Distributions and Interactions', fontsize=25, y=1.08)

plt.show()


## === cell 11

g = sns.pairplot(df_train[['FVC', 'Weeks', 'Percent', 'Age', 'Sex']], hue='Sex', aspect=1.4, height=5, diag_kind='kde', kind='reg')

g.axes[3, 0].set_xlabel('FVC', fontsize=20)
g.axes[3, 1].set_xlabel('Weeks', fontsize=20)
g.axes[3, 2].set_xlabel('Percent', fontsize=20)
g.axes[3, 3].set_xlabel('Age', fontsize=20)
g.axes[0, 0].set_ylabel('FVC', fontsize=20)
g.axes[1, 0].set_ylabel('Weeks', fontsize=20)
g.axes[2, 0].set_ylabel('Percent', fontsize=20)
g.axes[3, 0].set_ylabel('Age', fontsize=20)

g.axes[3, 0].tick_params(axis='x', labelsize=15)
g.axes[3, 1].tick_params(axis='x', labelsize=15)
g.axes[3, 2].tick_params(axis='x', labelsize=15)
g.axes[3, 3].tick_params(axis='x', labelsize=15)
g.axes[0, 0].tick_params(axis='y', labelsize=15)
g.axes[1, 0].tick_params(axis='y', labelsize=15)
g.axes[2, 0].tick_params(axis='y', labelsize=15)
g.axes[3, 0].tick_params(axis='y', labelsize=15)

plt.legend(prop={'size': 20})
g._legend.remove()
g.fig.suptitle('Tabular Data Feature Distributions and Interactions Between Sex Groups', fontsize=25, y=1.08)

plt.show()


## === cell 12

g = sns.pairplot(df_train[['FVC', 'Weeks', 'Percent', 'Age', 'SmokingStatus']], hue='SmokingStatus', aspect=1.4, height=5, diag_kind='kde', kind='reg')

g.axes[3, 0].set_xlabel('FVC', fontsize=20)
g.axes[3, 1].set_xlabel('Weeks', fontsize=20)
g.axes[3, 2].set_xlabel('Percent', fontsize=20)
g.axes[3, 3].set_xlabel('Age', fontsize=20)
g.axes[0, 0].set_ylabel('FVC', fontsize=20)
g.axes[1, 0].set_ylabel('Weeks', fontsize=20)
g.axes[2, 0].set_ylabel('Percent', fontsize=20)
g.axes[3, 0].set_ylabel('Age', fontsize=20)

g.axes[3, 0].tick_params(axis='x', labelsize=15)
g.axes[3, 1].tick_params(axis='x', labelsize=15)
g.axes[3, 2].tick_params(axis='x', labelsize=15)
g.axes[3, 3].tick_params(axis='x', labelsize=15)
g.axes[0, 0].tick_params(axis='y', labelsize=15)
g.axes[1, 0].tick_params(axis='y', labelsize=15)
g.axes[2, 0].tick_params(axis='y', labelsize=15)
g.axes[3, 0].tick_params(axis='y', labelsize=15)

plt.legend(prop={'size': 20})
g._legend.remove()
g.fig.suptitle('Tabular Data Feature Distributions and Interactions Between SmokingStatus Groups', fontsize=25, y=1.08)

plt.show()


## === cell 13
fig = plt.figure(figsize=(10, 10), dpi=100)

sns.heatmap(df_train.corr(), annot=True, square=True, cmap='coolwarm', annot_kws={'size': 15},  fmt='.2f')   

plt.tick_params(axis='x', labelsize=18, rotation=75)
plt.tick_params(axis='y', labelsize=18, rotation=0)
plt.title('Tabular Data Feature Correlations', size=20, pad=20)

plt.show()


## --- ERROR in cell 13, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3519592426.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0mfig[0m [0;34m=[0m [0mplt[0m[0;34m.[0m[0mfigure[0m[0;34m([0m[0mfigsize[0m[0;34m=[0m[0;34m([0m[0;36m10[0m[0;34m,[0m [0;36m10[0m[0;34m)[0m[0;34m,[0m [0mdpi[0m[0;34m=[0m[0;36m100[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m [0msns[0m[0;34m.[0m[0mheatmap[0m[0;34m([0m[0mdf_train[0m[0;34m.[0m[0mcorr[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mannot[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0msquare[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0mcmap[0m[0;34m=[0m[0;34m'coolwarm'[0m[0;34m,[0m [0mannot_kws[0m[0;34m=[0m[0;34m{[0m[0;34m'size'[0m[0;34m:[0m [0;36m15[0m[0;34m}[0m[0;34m,[0m  [0mfmt[0m[0;34m=[0m[0;34m'.2f'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0;34m[0m[0m
[1;32m      6[0m [0mplt[0m[0;34m.[0m[0mtick_params[0m[0;34m([0m[0maxis[0m[0;34m=[0m[0;34m'x'[0m[0;34m,[0m [0mlabelsize[0m[0;34m=[0m[0;36m18[0m[0;34m,[0m [0mrotation[0m[0;34m=[0m[0;36m75[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

[0;31mValueError[0m: could not convert string to float: 'ID00133637202223847701934'

## === cell 14
import os
import numpy as np
import pandas as pd
import random
import math

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import mean_squared_error
import category_encoders as ce

from sklearn.linear_model import Ridge, ElasticNet
from functools import partial
import scipy as sp

import warnings
warnings.filterwarnings("ignore")
