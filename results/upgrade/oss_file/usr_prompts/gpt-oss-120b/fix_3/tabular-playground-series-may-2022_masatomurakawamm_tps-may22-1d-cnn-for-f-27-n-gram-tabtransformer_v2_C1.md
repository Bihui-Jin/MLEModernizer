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
tqdm==4.67.1
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

0.93653

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    print("Seeds setted!")


global_seed = 42
seed_all(global_seed)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3967924431.py in <cell line: 0>()
      7 
      8 global_seed = 42
----> 9 seed_all(global_seed)
     10 

/tmp/ipykernel_11/3967924431.py in seed_all(s)
      1 def seed_all(s):
----> 2     random.seed(s)
      3     np.random.seed(s)
      4     os.environ["PYTHONHASHSEED"] = str(s)
      5     print("Seeds setted!")

NameError: name 'random' is not defined

## === cell 1
data_config = {
    "train_csv_path": "../input/tabular-playground-series-may-2022/train.csv",
    "test_csv_path": "../input/tabular-playground-series-may-2022/test.csv",
    "sample_submission_path": "../input/tabular-playground-series-may-2022/sample_submission.csv",
}

train_df = pd.read_csv(data_config["train_csv_path"])
test_df = pd.read_csv(data_config["test_csv_path"])
submission_df = pd.read_csv(data_config["sample_submission_path"])

print(f"train_length: {len(train_df)}")
print(f"test_lenght: {len(test_df)}")
print(f"submission_length: {len(submission_df)}")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1543501606.py in <cell line: 0>()
      5 }
      6 
----> 7 train_df = pd.read_csv(data_config["train_csv_path"])
      8 test_df = pd.read_csv(data_config["test_csv_path"])
      9 submission_df = pd.read_csv(data_config["sample_submission_path"])

NameError: name 'pd' is not defined

## === cell 2
train_df.head()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/577764774.py in <cell line: 0>()
----> 1 train_df.head()
      2 

NameError: name 'train_df' is not defined

## === cell 3
print("train_df.info()")
print(train_df.info(), "\n")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2626682488.py in <cell line: 0>()
      1 print("train_df.info()")
----> 2 print(train_df.info(), "\n")
      3 

NameError: name 'train_df' is not defined

## === cell 4
numeric_corr = train_df.select_dtypes(include="number").corr()
fig = px.imshow(
    numeric_corr,
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
    aspect="auto",
)
fig.update_layout(height=750, title="Heatmap", showlegend=False)
fig.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2850424724.py in <cell line: 0>()
----> 1 numeric_corr = train_df.select_dtypes(include="number").corr()
      2 fig = px.imshow(
      3     numeric_corr,
      4     color_continuous_scale="RdBu_r",
      5     color_continuous_midpoint=0,

NameError: name 'train_df' is not defined

## === cell 5
target_count = train_df.groupby(["target"])["id"].count()
target_percent = target_count / target_count.sum()

fig = go.Figure()
data = go.Bar(x=target_count.index.astype(str).values, y=target_count.values)
fig.add_trace(data)
fig.update_layout(
    title=dict(text="target distribution"),
    xaxis=dict(title="target values"),
    yaxis=dict(title="counts"),
)
fig.show()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/343274051.py in <cell line: 0>()
----> 1 target_count = train_df.groupby(["target"])["id"].count()
      2 target_percent = target_count / target_count.sum()
      3 
      4 fig = go.Figure()
      5 data = go.Bar(x=target_count.index.astype(str).values, y=target_count.values)

NameError: name 'train_df' is not defined

## === cell 6
train_pos_df = train_df.query("target==1")
train_neg_df = train_df.query("target==0")

numerical_columns = [
    "f_00",
    "f_01",
    "f_02",
    "f_03",
    "f_04",
    "f_05",
    "f_06",
    "f_19",
    "f_20",
    "f_21",
    "f_22",
    "f_23",
    "f_24",
    "f_25",
    "f_26",
    "f_28",
]
categorical_columns = [
    "f_07",
    "f_08",
    "f_09",
    "f_10",
    "f_11",
    "f_12",
    "f_13",
    "f_14",
    "f_15",
    "f_16",
    "f_17",
    "f_18",
    "f_29",
    "f_30",
]
obj_columns = ["f_27"]

print(
    f"numerical_columns: {len(numerical_columns)},  categorical_columns: {len(categorical_columns)},  obj_columns: {len(obj_columns)}"
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1417522828.py in <cell line: 0>()
----> 1 train_pos_df = train_df.query("target==1")
      2 train_neg_df = train_df.query("target==0")
      3 
      4 numerical_columns = [
      5     "f_00",

NameError: name 'train_df' is not defined

## === cell 7
train_df[numerical_columns].describe()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2991361170.py in <cell line: 0>()
----> 1 train_df[numerical_columns].describe()
      2 

NameError: name 'train_df' is not defined

## === cell 8
fig = plt.figure(figsize=(16, 10))
for i, c in enumerate(numerical_columns):
    ax = fig.add_subplot(4, 4, i + 1)
    ax.hist(train_pos_df[c], color="b", alpha=0.5, bins=50)
    ax.hist(train_neg_df[c], color="r", alpha=0.5, bins=50)
    ax.set_title(c)
fig.suptitle(
    'Distributions of Numerical Features (Blue: "target=1", red: "target=0")',
    fontsize=20,
)
fig.tight_layout()
plt.show()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/703445835.py in <cell line: 0>()
----> 1 fig = plt.figure(figsize=(16, 10))
      2 for i, c in enumerate(numerical_columns):
      3     ax = fig.add_subplot(4, 4, i + 1)
      4     ax.hist(train_pos_df[c], color="b", alpha=0.5, bins=50)
      5     ax.hist(train_neg_df[c], color="r", alpha=0.5, bins=50)

NameError: name 'plt' is not defined

## === cell 9
train_df[categorical_columns].describe()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3271514419.py in <cell line: 0>()
----> 1 train_df[categorical_columns].describe()
      2 

NameError: name 'train_df' is not defined

## === cell 10
fig = plt.figure(figsize=(16, 10))
for i, c in enumerate(categorical_columns):
    ax = fig.add_subplot(4, 4, i + 1)
    x_range = (train_df[c].min(), train_df[c].max())
    ax.hist(train_pos_df[c], color="b", alpha=0.5, range=x_range)
    ax.hist(train_neg_df[c], color="r", alpha=0.5, range=x_range)
    ax.set_title(c)
fig.suptitle(
    'Distributions of Categorical Features (Blue: "target=1", red: "target=0")',
    fontsize=20,
)
fig.tight_layout()
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2786259742.py in <cell line: 0>()
----> 1 fig = plt.figure(figsize=(16, 10))
      2 for i, c in enumerate(categorical_columns):
      3     ax = fig.add_subplot(4, 4, i + 1)
      4     x_range = (train_df[c].min(), train_df[c].max())
      5     ax.hist(train_pos_df[c], color="b", alpha=0.5, range=x_range)

NameError: name 'plt' is not defined

## === cell 11
f_27_df = train_df[["f_27", "target"]]
f_27_feature_df = f_27_df.drop(["f_27"], axis=1)
f_27_feature_df["n_char"] = f_27_df["f_27"].map(lambda x: len(x))

for i in range(65, 91):  # ASCII A-Z
    f_27_feature_df[chr(i)] = f_27_df["f_27"].map(lambda x: x.count(chr(i)))

f_27_feature_df.describe()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/972075743.py in <cell line: 0>()
----> 1 f_27_df = train_df[["f_27", "target"]]
      2 f_27_feature_df = f_27_df.drop(["f_27"], axis=1)
      3 f_27_feature_df["n_char"] = f_27_df["f_27"].map(lambda x: len(x))
      4 
      5 for i in range(65, 91):  # ASCII A-Z

NameError: name 'train_df' is not defined

## === cell 12
tmp_df = f_27_feature_df.groupby(["target"]).sum()
tmp_df = tmp_df.drop(["n_char"], axis=1)

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=["target=0", "target=1"],
    shared_xaxes="all",
    shared_yaxes="all",
)
for row in range(2):
    data = go.Bar(
        x=tmp_df.columns.astype(str).values, y=tmp_df.iloc[row].values.squeeze()
    )
    fig.add_trace(data, row=row + 1, col=1)
fig.update_layout(title="Count of Characters", showlegend=False)
fig.show()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/657014221.py in <cell line: 0>()
----> 1 tmp_df = f_27_feature_df.groupby(["target"]).sum()
      2 tmp_df = tmp_df.drop(["n_char"], axis=1)
      3 
      4 fig = make_subplots(
      5     rows=2,

NameError: name 'f_27_feature_df' is not defined

## === cell 13
train_df = train_df.drop(["f_27"], axis=1)
f_27_feature_df = f_27_feature_df.drop(["target"], axis=1)
train = pd.merge(train_df, f_27_feature_df, left_index=True, right_index=True)

obj_cols = train.select_dtypes(include="object").columns
for col in obj_cols:
    train[col] = train[col].astype("category").cat.codes



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/893137389.py in <cell line: 0>()
----> 1 train_df = train_df.drop(["f_27"], axis=1)
      2 f_27_feature_df = f_27_feature_df.drop(["target"], axis=1)
      3 train = pd.merge(train_df, f_27_feature_df, left_index=True, right_index=True)
      4 
      5 obj_cols = train.select_dtypes(include="object").columns

NameError: name 'train_df' is not defined

## === cell 14
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=global_seed)
train["k_folds"] = -1
for fold, (train_idx, valid_idx) in enumerate(skf.split(X=train, y=train["target"])):
    train.loc[valid_idx, "k_folds"] = fold

models = []
for fold in range(n_splits):
    print(f"======fold {fold}======")
    valid_tmp = train[train["k_folds"] == fold]
    train_tmp = train[train["k_folds"] != fold]

    y_train = train_tmp["target"]
    X_train = train_tmp.drop(["id", "target", "k_folds"], axis=1)

    y_valid = valid_tmp["target"]
    X_valid = valid_tmp.drop(["id", "target", "k_folds"], axis=1)

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",  # use CPU histogram implementation
        seed=global_seed,
        eval_metric="auc",
        use_label_encoder=False,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=10,
        verbose=False,
    )
    models.append(model)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2333035164.py in <cell line: 0>()
      1 n_splits = 5
----> 2 skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=global_seed)
      3 train["k_folds"] = -1
      4 for fold, (train_idx, valid_idx) in enumerate(skf.split(X=train, y=train["target"])):
      5     train.loc[valid_idx, "k_folds"] = fold

NameError: name 'StratifiedKFold' is not defined

## === cell 15
fig, ax = plt.subplots(1, 1, figsize=(20, 12))
plot_importance(models[-1], ax=ax, xlabel=None)
plt.title("XGB Feature importance", fontsize=20)
plt.show()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1844246942.py in <cell line: 0>()
----> 1 fig, ax = plt.subplots(1, 1, figsize=(20, 12))
      2 plot_importance(models[-1], ax=ax, xlabel=None)
      3 plt.title("XGB Feature importance", fontsize=20)
      4 plt.show()
      5 

NameError: name 'plt' is not defined

## === cell 16
test_df["n_char"] = test_df["f_27"].map(lambda x: len(x))
for i in range(65, 91):  # ASCII A-Z
    test_df[chr(i)] = test_df["f_27"].map(lambda x: x.count(chr(i)))

test_df = test_df.drop(["id", "f_27"], axis=1)

obj_test_cols = test_df.select_dtypes(include="object").columns
for col in obj_test_cols:
    test_df[col] = test_df[col].astype("category").cat.codes



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3019089155.py in <cell line: 0>()
      7 # the XGBoost “feature_names mismatch” error.
      8 # -------------------------------------------------------------------------
----> 9 test_df["n_char"] = test_df["f_27"].map(lambda x: len(x))
     10 for i in range(65, 91):  # ASCII A-Z
     11     test_df[chr(i)] = test_df["f_27"].map(lambda x: x.count(chr(i)))

NameError: name 'test_df' is not defined

## === cell 17
for i, model in enumerate(models):
    submission_df[f"pred_{i}"] = model.predict_proba(test_df)[:, 1]
submission_df.head()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1023799110.py in <cell line: 0>()
----> 1 for i, model in enumerate(models):
      2     submission_df[f"pred_{i}"] = model.predict_proba(test_df)[:, 1]
      3 submission_df.head()
      4 

NameError: name 'models' is not defined

## === cell 18
pred_cols = [f"pred_{i}" for i in range(len(models))]
submission_df["target"] = submission_df[pred_cols].mean(axis=1)
submission_df = submission_df.drop(columns=pred_cols)
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
submission_df.head()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1653034817.py in <cell line: 0>()
----> 1 pred_cols = [f"pred_{i}" for i in range(len(models))]
      2 submission_df["target"] = submission_df[pred_cols].mean(axis=1)
      3 submission_df = submission_df.drop(columns=pred_cols)
      4 submission_df.to_csv("submission.csv", index=False)
      5 print("Submission saved to submission.csv")

NameError: name 'models' is not defined
