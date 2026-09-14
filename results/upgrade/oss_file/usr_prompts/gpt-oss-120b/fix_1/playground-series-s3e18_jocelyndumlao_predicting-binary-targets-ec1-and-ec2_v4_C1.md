# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict values for synthetic data.

### Description
## Metric
Area under the ROC curve for each target, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict the value for the targets `EC1` and `EC2`. The file should contain a header and have the following format:

```
id,EC1,EC2
14838,0.22,0.71
14839,0.78,0.43
14840,0.53,0.11
etc.
```

## Dataset 
- **train.csv** - the training dataset; `[EC1 - EC6]` are the (binary) targets, although you are only asked to predict `EC1` and `EC2`.
- **test.csv** - the test dataset; your objective is to predict the probability of the two targets `EC1` and `EC2`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        input/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        working/
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
```

-> data/playground-series-s3e18/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/playground-series-s3e18/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/playground-series-s3e18/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> data/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.62651

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import matplotlib.pyplot as plt

substrates = ['Substrate A', 'Substrate B', 'Substrate C', 'Substrate D']
labels = [['Enzyme 1', 'Enzyme 2'], ['Enzyme 2'], ['Enzyme 1', 'Enzyme 3'], ['Enzyme 2', 'Enzyme 3']]

fig, ax = plt.subplots()

for i, substrate in enumerate(substrates):
    for label in labels[i]:
        ax.scatter(i, label, color='blue')
        ax.annotate(label, (i, label))

ax.set_xticks(range(len(substrates)))
ax.set_xticklabels(substrates)
ax.set_ylabel('Enzymatic Activities')
ax.set_title('Multi-label Classification of Enzyme Substrates')

plt.show()


## === cell 1
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")


## === cell 2
train_df = pd.read_csv('/kaggle/input/playground-series-s3e18/train.csv')
train_df.head().style.set_properties(**{'background-color':'pink','color':'black','border-color':'#8b8c8c'})


## === cell 3
test_df = pd.read_csv('/kaggle/input/playground-series-s3e18/test.csv')
test_df.head().style.set_properties(**{'background-color':'lightblue','color':'black','border-color':'#8b8c8c'})


## === cell 4
submission = pd.read_csv('/kaggle/input/playground-series-s3e18/sample_submission.csv')
submission.head().style.set_properties(**{'background-color':'green','color':'black','border-color':'#8b8c8c'})


## === cell 5
train_df.dropna(inplace=True)


## === cell 6
train_df.isnull().sum()


## === cell 7
train_df.info()
test_df.info()
submission.info()


## === cell 8
train_df.drop_duplicates()


## === cell 9
print("Shape of train:", train_df.shape)
print("Shape of test:", test_df.shape)
print("Shape of submission:", submission.shape)


## === cell 10
print("Number of duplicates in training dataset:", train_df.duplicated().sum())
print("Number of duplicates in testing dataset:", test_df.duplicated().sum())
print("Number of duplicates in submission dataset:", submission.duplicated().sum())


## === cell 11
train_df.columns


## === cell 12
styled_data = train_df.describe().style\
.background_gradient(cmap='coolwarm')\
.set_properties(**{'text-align':'center','border':'1px solid black'})

display(styled_data)


## === cell 13
import plotly.express as px

counts = train_df['EC1'].value_counts()
fig = px.bar(x=counts.index, y=counts.values, color=['Yes', 'No'],
             color_discrete_sequence=['#1f77b4', '#ff7f0e'])
fig.update_layout(title='Bar Plot - EC1', xaxis_title='EC1', yaxis_title='Count', showlegend=False)
fig.show()


## === cell 14
counts = train_df['EC2'].value_counts()
fig = px.bar(x=counts.index, y=counts.values, color=['Yes', 'No'],
             color_discrete_sequence=['#1f77b4', '#ff7f0e'])
fig.update_layout(title='Bar Plot - EC2', xaxis_title='EC2', yaxis_title='Count', showlegend=False)
fig.show()


## === cell 15
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

column_names = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v', 'Chi4n', 'EState_VSA1', 'EState_VSA2',
                'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
                'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex', 'NumHeteroatoms', 'PEOE_VSA10',
                'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3',
                'VSA_EState9', 'fr_COO']

for i, column in enumerate(column_names):
    fig = px.histogram(train_df, x=column, nbins=10, marginal='box',
                       labels={column: column})
    fig.update_layout(title='Histogram', xaxis_title=column, yaxis_title='Count',
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(size=12))
    fig.update_traces(marker_color=colors[i % len(colors)])
    fig.show()
