# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

# 3. Installed packages

catboost==1.2.8
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor




## === cell 2
def df_overview(df):
    print("Dataframe overview:\n")
    display(df.head())
    print("--------------------------------------------\nSample:\n")
    display(df.sample(10, random_state=555))
    print("--------------------------------------------\nInfo:\n")
    print(df.info())
    print("--------------------------------------------\nNaN's:\n")
    print(df.isna().sum())
    print("--------------------------------------------\nDescribe:\n")
    display(df.describe())
    print("--------------------------------------------\nFeature correlation:\n")
    display(df.corr())




## === cell 3
def show_correlogram(df):
    plt.figure(figsize=(6, 6), dpi=80)
    sns.heatmap(
        df.corr(),
        xticklabels=df.corr().columns,
        yticklabels=df.corr().columns,
        cmap="RdYlGn",
        center=0,
        annot=True,
        cbar=False,
    )
    plt.title("Correlogram between features", fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()




## === cell 4
def plot_create(x, y):
    plt.plot(x, y, "-", label=y.name)


def process_visualisation(df, breath_id):
    plt.figure(figsize=(14, 6))
    plt.title("Breath Id - {}".format(breath_id))
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["pressure"],
    )
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["u_in"],
    )
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["u_out"],
    )
    plt.grid()
    plt.legend()
    plt.ylabel("Value")
    plt.show()




## === cell 5
def process_visualisation_with_preds(df, df_preds, breath_id):
    plt.figure(figsize=(14, 6))
    plt.title("Breath Id - {}".format(breath_id))
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["pressure"],
    )
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["u_in"],
    )
    plot_create(
        df[df["breath_id"] == breath_id]["time_step"],
        df[df["breath_id"] == breath_id]["u_out"],
    )
    plot_create(df[df["breath_id"] == breath_id]["time_step"], df_preds)
    plt.grid()
    plt.legend()
    plt.ylabel("Value")
    plt.show()




## === cell 6
def add_features(df):
    df["u_in_cumsum"] = df["u_in"].groupby(df["breath_id"]).cumsum()
    df["u_in_lag_1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_in_lag_2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_in_rolling_mean"] = df.groupby("breath_id")["u_in"].shift().rolling(3).mean()
    df["u_in_begin"] = df.groupby("breath_id")["u_in"].transform("first")
    df["u_in_end"] = df.groupby("breath_id")["u_in"].transform("last")
    df["u_in_min"] = df.groupby("breath_id")["u_in"].transform("min")
    df["u_in_max"] = df.groupby("breath_id")["u_in"].transform("max")
    df["u_in_median"] = df.groupby("breath_id")["u_in"].transform("median")
    df = df.fillna(0)
    df = df.drop(columns=["id"], errors="ignore")
    return df




## === cell 7
def train_and_score(model):
    if isinstance(model, CatBoostRegressor):
        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
            eval_set=(X_valid, y_valid),
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)
    return mean_absolute_error(y_valid, model.predict(X_valid))




## === cell 8
df_train = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/train.csv")
df_test = pd.read_csv("/kaggle/input/ventilator-pressure-prediction/test.csv")
df_sample_submission = pd.read_csv(
    "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
)



## === cell 9
df_train = df_train.drop("id", axis=1)



## === cell 10
df_overview(df_train)



## === cell 11
df_overview(df_test)



## === cell 12
show_correlogram(df_train)



## === cell 13
for col in df_train.columns:
    df_train[col].plot(kind="hist", bins=30, title=col)
    plt.show()



## === cell 14
for col in ["time_step", "u_in", "pressure"]:
    df_train[col].plot(kind="box")
    plt.show()



## === cell 15
df_train[df_train["pressure"] > 55]["pressure"].plot(
    kind="hist", bins=30, title="Pressure > 55"
)
plt.show()
df_train[df_train["u_in"] > 70]["u_in"].plot(kind="hist", bins=30, title="U in > 70")
plt.show()



## === cell 16
df_train[df_train["pressure"] > 64.5]["pressure"].value_counts().sort_index(
    ascending=False
)



## === cell 17
df_train[df_train["u_in"] > 99.98]["u_in"].value_counts().sort_index(ascending=False)



## === cell 18
df_test[df_test["u_in"] > 99.98]["u_in"].value_counts().sort_index(ascending=False)



## === cell 19
df_train[df_train["u_in"] > 99.98]["pressure"].plot(
    kind="hist", bins=40, title="Pressure when U in is high"
)
plt.show()
df_train[df_train["pressure"] > 64.5]["u_in"].plot(
    kind="hist", bins=40, title="U in when Pressure is high"
)
plt.show()



## === cell 20
df_breath = df_train.groupby("breath_id", as_index=False).median()
df_breath



## === cell 21
for col in ["time_step", "u_in", "pressure"]:
    df_breath[col].plot(kind="hist", bins=30, title=col)
    plt.show()



## === cell 22
df_breath[df_breath["pressure"] < 0].head(10)



## === cell 23
df_breath[df_breath["pressure"] < 0]["pressure"].plot(
    kind="hist", bins=30, title="Negative pressure"
)
plt.show()



## === cell 24
df_breath[df_breath["pressure"] < 0]["u_in"].plot(
    kind="hist", bins=30, title="U in when Pressure < 0"
)
plt.show()
df_breath[df_breath["pressure"] == df_breath["pressure"].median()]["u_in"].plot(
    kind="hist", bins=30, title="U in when Pressure is normal"
)
plt.show()



## === cell 25
print("Some data where pressure is normal:")
display(
    df_breath[df_breath["pressure"] == df_train["pressure"].median()].sample(
        3, random_state=1
    )
)
print("\nSome data where pressure is below 0:")
display(df_breath[df_breath["pressure"] < 0].sample(3, random_state=1))
print("\nSome data where pressure is high:")
display(df_breath[df_breath["pressure"] > 12].sample(3, random_state=1))



## === cell 26
print("Process visualisation where pressure is normal:")
process_visualisation(df_train, 48945)
process_visualisation(df_train, 28141)
process_visualisation(df_train, 109737)



## === cell 27
print("Process visualisation where pressure is below 0:")
process_visualisation(df_train, 98041)
process_visualisation(df_train, 118131)
process_visualisation(df_train, 11216)



## === cell 28
print("Process visualisation where pressure is high:")
process_visualisation(df_train, 104581)
process_visualisation(df_train, 14416)
process_visualisation(df_train, 69384)



## === cell 29
X = df_train.copy()
X = X.drop("pressure", axis=1)
X = add_features(X)
y = df_train["pressure"]



## === cell 30
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=555
)
cat_features = ["breath_id"]



## === cell 31
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(max_depth=15, random_state=555)
cb_model = CatBoostRegressor(
    iterations=500,
    depth=10,
    learning_rate=0.05,
    loss_function="MAE",
    random_seed=555,
    verbose=0,
    thread_count=4,
)
dummy = DummyRegressor()



## === cell 32
display(
    pd.DataFrame(
        data=(
            [train_and_score(linear_model)],
            [train_and_score(tree_model)],
            [train_and_score(cb_model)],
            [train_and_score(dummy)],
        ),
        columns=["Result MAE"],
        index=["Linear", "Tree", "CatBoost", "Dummy"],
    )
)



## === cell 33
pd.DataFrame(
    cb_model.get_feature_importance(),
    index=cb_model.feature_names_,
    columns=["importances"],
).sort_values(by="importances").plot(
    kind="barh", figsize=(8, 6), title="CatBoost feature importances"
)
plt.show()



## === cell 34
linear_model.fit(X, y)
tree_model.fit(X, y)
cb_model.fit(X, y, cat_features=cat_features)



## === cell 35
X_df_vis = df_train[df_train["breath_id"] == 28141].reset_index()
X_df_vis = add_features(X_df_vis)
X_df_vis = X_df_vis.drop(["index", "pressure"], axis=1)

print("Pressure predictions by Linear Model where pressure is normal:")
process_visualisation_with_preds(
    df_train, pd.Series(linear_model.predict(X_df_vis), name="predictions"), 28141
)
print("Pressure predictions by Tree Model where pressure is normal:")
process_visualisation_with_preds(
    df_train, pd.Series(tree_model.predict(X_df_vis), name="predictions"), 28141
)
print("Pressure predictions by CatBoost Model where pressure is normal:")
process_visualisation_with_preds(
    df_train, pd.Series(cb_model.predict(X_df_vis), name="predictions"), 28141
)



## === cell 36
X_df_vis = df_train[df_train["breath_id"] == 98041].reset_index()
X_df_vis = add_features(X_df_vis)
X_df_vis = X_df_vis.drop(["index", "pressure"], axis=1)

print("Pressure predictions by Linear Model where pressure is below 0:")
process_visualisation_with_preds(
    df_train, pd.Series(linear_model.predict(X_df_vis), name="predictions"), 98041
)
print("Pressure predictions by Tree Model where pressure is below 0:")
process_visualisation_with_preds(
    df_train, pd.Series(tree_model.predict(X_df_vis), name="predictions"), 98041
)
print("Pressure predictions by CatBoost Model where pressure is below 0:")
process_visualisation_with_preds(
    df_train, pd.Series(cb_model.predict(X_df_vis), name="predictions"), 98041
)



## === cell 37
X_df_vis = df_train[df_train["breath_id"] == 104581].reset_index()
X_df_vis = add_features(X_df_vis)
X_df_vis = X_df_vis.drop(["index", "pressure"], axis=1)

print("Pressure predictions by Linear Model where pressure is high:")
process_visualisation_with_preds(
    df_train, pd.Series(linear_model.predict(X_df_vis), name="predictions"), 104581
)
print("Pressure predictions by Tree Model where pressure is high:")
process_visualisation_with_preds(
    df_train, pd.Series(tree_model.predict(X_df_vis), name="predictions"), 104581
)
print("Pressure predictions by CatBoost Model where pressure is high:")
process_visualisation_with_preds(
    df_train, pd.Series(cb_model.predict(X_df_vis), name="predictions"), 104581
)



## === cell 38
df_test_featured = df_test.copy()
df_test_featured = add_features(df_test_featured)
df_test_featured = df_test_featured.drop(columns=["id"], errors="ignore")



## === cell 39
preds = cb_model.predict(df_test_featured)



## === cell 40
output = pd.DataFrame({"id": df_test["id"].values, "pressure": preds})
display(output.head())
output.to_csv("submission.csv", index=False)
