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

0.93476

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings
import random
import os
import gc
import datetime

import numpy as np
import pandas as pd

import torch
import sklearn.exceptions

import matplotlib

matplotlib.use("Agg")  # avoid backend issues in Kaggle non-interactive runs
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    MaxAbsScaler,
)
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

try:
    import shap

    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")




## === cell 1
def jupyter_setting():
    pd.options.display.max_columns = None

    warnings.filterwarnings(action="ignore")
    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings(
        "ignore", category=sklearn.exceptions.UndefinedMetricWarning
    )

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.max_colwidth", None)

    icecream = ["#00008b", "#960018", "#008b00", "#00468b", "#8b4500", "#582c00"]

    colors = [
        "lightcoral",
        "sandybrown",
        "darkorange",
        "mediumseagreen",
        "lightseagreen",
        "cornflowerblue",
        "mediumpurple",
        "palevioletred",
        "lightskyblue",
        "sandybrown",
        "yellowgreen",
        "indianred",
        "lightsteelblue",
        "mediumorchid",
        "deepskyblue",
    ]

    myred = "#CD5C5C"
    myblue = "#6495ED"
    mygreen = "#90EE90"
    color_cols = [myred, myblue, mygreen]

    return icecream, colors, color_cols


icecream, colors, color_cols = jupyter_setting()




## === cell 2
def missing_zero_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = round(df.isnull().mean().mul(100), 2)
    mz_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
        columns={
            df.index.name: "col_name",
            0: "Valores ausentes",
            1: "% de valores totais",
        }
    )

    mz_table["Tipo de dados"] = df.dtypes
    mz_table = mz_table[mz_table.iloc[:, 1] != 0].sort_values(
        "% de valores totais", ascending=False
    )

    msg = "Seu dataframe selecionado tem {} colunas e {} linhas. \nExistem {} colunas com valores ausentes."
    print(msg.format(df.shape[1], df.shape[0], mz_table.shape[0]))

    return mz_table.reset_index()




## === cell 3
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

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
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / max(start_mem, 1e-9)
            )
        )
    return df




## === cell 4
def diff(t_a, t_b):
    from dateutil.relativedelta import relativedelta

    t_diff = relativedelta(t_b, t_a)
    return "{h}h {m}m {s}s".format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


def free_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




## === cell 5
def feature_engineering(df_):
    var_f27 = ""
    for col in df_["f_27"].astype(str).values:
        var_f27 += col

    var_f27 = sorted(list(set(var_f27)))

    df_ = df_.copy()
    df_["fe_f_27_unique"] = df_["f_27"].astype(str).apply(lambda x: len(set(x)))

    for letra in var_f27:
        df_["fe_" + letra.lower() + "_count"] = df_["f_27"].astype(str).str.count(letra)

    return df_




## === cell 6
COLAB = False
path_data = ""
target = "target"

_candidate_paths = [
    "/kaggle/input/tabular-playground-series-may-2022/",
    "/kaggle/data/tabular-playground-series-may-2022/",
    "../input/tabular-playground-series-may-2022/",
]
path = None
for p in _candidate_paths:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        path = p
        break
if path is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv in any known input paths: "
        + ", ".join(_candidate_paths)
    )

paths = [
    "img",
    "Data",
    "Data/pkl",
    "Data/submission",
    "Data/tunning",
    "model",
    "model/preds",
    "model/optuna",
    "model/preds/test",
    "model/preds/test/n1",
    "model/preds/test/n2",
    "model/preds/test/n3",
    "model/preds/train",
    "model/preds/train/n1",
    "model/preds/train/n2",
    "model/preds/train/n3",
    "model/preds/param",
]
for p in paths:
    os.makedirs(p, exist_ok=True)

print("Using input path:", path)




## === cell 7
df1_train = pd.read_csv(path + path_data + "train.csv")
df1_test = pd.read_csv(path + path_data + "test.csv")
df_submission = pd.read_csv(path + path_data + "sample_submission.csv")

print(df1_train.shape, df1_test.shape, df_submission.shape)




## === cell 8
df1_train = reduce_memory_usage(df1_train, verbose=True)
df1_test = reduce_memory_usage(df1_test, verbose=True)

df2_train = df1_train.copy()
df2_test = df1_test.copy()

print("TREINO")
print("Number of Rows: {}".format(df2_train.shape[0]))
print("Number of Columns: {}".format(df2_train.shape[1]), end="\n\n")

print("TESTE")
print("Number of Rows: {}".format(df2_test.shape[0]))
print("Number of Columns: {}".format(df2_test.shape[1]))




## === cell 9
df2_train = feature_engineering(df2_train)
df2_test = feature_engineering(df2_test)

feature_float = df2_test.select_dtypes(np.number).columns.to_list()
feature_cat = df2_test.select_dtypes(object).columns.to_list()
if "id" in feature_float:
    feature_float.remove("id")

print(
    f"Temos {len(feature_float)} variávies numéricas e {len(feature_cat)} categóricas."
)




## === cell 10
if target in df2_train.columns:
    int8_cols = df2_train.select_dtypes(np.int8).columns
    int8_cols = [c for c in int8_cols if c != target]
else:
    int8_cols = df2_train.select_dtypes(np.int8).columns.tolist()

for col in int8_cols:
    df2_train[col] = df2_train[col].astype(object)
    df2_test[col] = df2_test[col].astype(object)

df3_train = df2_train.copy()
df3_test = df2_test.copy()

if "f_27" in df3_train.columns:
    df3_train.drop(["f_27"], axis=1, inplace=True)
if "f_27" in df3_test.columns:
    df3_test.drop(["f_27"], axis=1, inplace=True)

for col in df3_train.select_dtypes(object).columns:
    df3_train[col] = df3_train[col].astype(np.int32)
    df3_test[col] = df3_test[col].astype(np.int32)

X = df3_train.drop([target, "id"], axis=1)
y = df3_train[target]
X_test = df3_test.drop(["id"], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, shuffle=True, stratify=y, random_state=12359
)

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape, X_test.shape)




## === cell 11
X_valid2, X_test_final, y_valid2, y_test_final = train_test_split(
    X_valid,
    y_valid,
    test_size=0.3,
    shuffle=True,
    stratify=y_valid,
    random_state=12359,
)
print(X_valid2.shape, y_valid2.shape, X_test_final.shape, y_test_final.shape)




## === cell 12
seed = 12359
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 1000,
    "random_state": seed,
}

if torch.cuda.is_available():
    params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})

params




## === cell 13
scalers = [
    StandardScaler(),
    RobustScaler(),
    MinMaxScaler(),
    MaxAbsScaler(),
    QuantileTransformer(output_distribution="normal", random_state=0),
]

cols = X_test.columns
scaler_best = None
model_best = None
auc_best = -1.0

for scaler in scalers:
    X_train_s = X_train.copy()
    X_valid_s = X_valid.copy()

    X_train_s = pd.DataFrame(scaler.fit_transform(X_train_s), columns=cols)
    X_valid_s = pd.DataFrame(scaler.transform(X_valid_s), columns=cols)

    model_tmp = xgb.XGBClassifier(**params)
    model_tmp.fit(X_train_s, y_train, verbose=False)

    y_pred_prob_vl = model_tmp.predict_proba(X_valid_s)[:, 1]
    auc_vl = metrics.roc_auc_score(y_valid, y_pred_prob_vl)

    print(f"AUC Val: {auc_vl:2.5f} => {scaler}")

    if auc_vl > auc_best:
        auc_best = auc_vl
        scaler_best = scaler
        model_best = model_tmp

print()
print("The Best")
print(f"Scaler: {scaler_best}")
print(f"AUC   : {auc_best:2.5f}")




## === cell 14
X_test_final_sc = pd.DataFrame(scaler_best.transform(X_test_final), columns=cols)
y_pred_prob = model_best.predict_proba(X_test_final_sc)[:, 1]
auc = metrics.roc_auc_score(y_test_final, y_pred_prob)
print("AUC dados não visto : {:2.5f}".format(auc))




## === cell 15
X_test_sc = pd.DataFrame(scaler_best.transform(X_test), columns=cols)
y_pred_ts = model_best.predict_proba(X_test_sc)[:, 1]

sub = df_submission.copy()
if "id" not in sub.columns:
    raise ValueError("sample_submission.csv must contain an 'id' column.")
if target not in sub.columns:
    sub[target] = 0.0

test_ids = df1_test["id"].values
pred_df = pd.DataFrame({"id": test_ids, target: y_pred_ts.astype(np.float64)})

sub = sub.drop(columns=[target]).merge(pred_df, on="id", how="left")
if sub[target].isna().any():
    raise ValueError("Submission has missing predictions after id-merge; id mismatch.")

sub_path = "Data/submission/submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote submission:", sub_path, "shape:", sub.shape)
print(sub.head())




## === cell 16
def cross_val_model(
    model_,
    model_name_,
    X_,
    y_,
    X_test_,
    target_,
    scalers_,
    lb_,
    fold_=5,
    path_="",
    seed_=12359,
    feature_scaler_=None,
    print_report_=False,
    save_submission_=False,
):
    n_estimators = model_.get_params()["n_estimators"]

    taco = 76
    acc_best = 0
    feature_imp = pd.DataFrame()
    model_list = []
    df_prob_temp_last = None

    for i, scaler_ in enumerate(scalers_):
        time_start = datetime.datetime.now()
        score = []

        if scaler_ is not None:
            string_scaler = str(scaler_)
            string_scaler = string_scaler[: string_scaler.index("(")]
        else:
            string_scaler = None

        folds = StratifiedKFold(n_splits=fold_, shuffle=True, random_state=seed_)

        print("=" * taco)
        print("Scaler: {} - n_estimators: {}".format(string_scaler, n_estimators))
        print("=" * taco)

        pred_test = 0.0

        for fold, (trn_idx, val_idx) in enumerate(folds.split(X_, y_)):
            time_fold_start = datetime.datetime.now()

            X_trn, X_val = X_.iloc[trn_idx].copy(), X_.iloc[val_idx].copy()
            y_trn, y_val = y_.iloc[trn_idx], y_.iloc[val_idx]

            X_tst = X_test_.copy()

            if scaler_ is not None:
                if feature_scaler_ is not None:
                    X_trn[feature_scaler_] = scaler_.fit_transform(
                        X_trn[feature_scaler_]
                    )
                    X_val[feature_scaler_] = scaler_.transform(X_val[feature_scaler_])
                    X_tst[feature_scaler_] = scaler_.transform(X_tst[feature_scaler_])
                else:
                    X_trn = scaler_.fit_transform(X_trn)
                    X_val = scaler_.transform(X_val)
                    X_tst = scaler_.transform(X_tst)

            model_fold = xgb.XGBClassifier(**model_.get_params())
            model_fold.fit(
                X_trn,
                y_trn,
                eval_set=[(X_trn, y_trn), (X_val, y_val)],
                early_stopping_rounds=int(n_estimators * 0.1),
                verbose=False,
            )

            best_it = getattr(model_fold, "best_iteration", None)
            if best_it is not None:
                iter_range = (0, int(best_it) + 1)
                y_pred_val_prob = model_fold.predict_proba(
                    X_val, iteration_range=iter_range
                )[:, 1]
                pred_test += (
                    model_fold.predict_proba(X_tst, iteration_range=iter_range)[:, 1]
                    / folds.n_splits
                )
            else:
                y_pred_val_prob = model_fold.predict_proba(X_val)[:, 1]
                pred_test += model_fold.predict_proba(X_tst)[:, 1] / folds.n_splits

            y_pred_val = (y_pred_val_prob > 0.5).astype(int)

            df_prob_temp = pd.DataFrame({"y_proba": y_pred_val_prob})
            df_prob_temp["fold"] = fold + 1
            df_prob_temp["id"] = val_idx
            df_prob_temp["y_val"] = y_val.values
            df_prob_temp["y_pred"] = y_pred_val
            df_prob_temp["scaler"] = str(string_scaler)
            df_prob_temp_last = df_prob_temp

            auc = metrics.roc_auc_score(y_val, y_pred_val_prob)
            f1 = metrics.f1_score(y_val, y_pred_val)
            ll = metrics.log_loss(y_val, y_pred_val)

            score.append(auc)

            feat_imp = pd.DataFrame(
                index=X_.columns,
                data=model_fold.feature_importances_,
                columns=["fold_{}".format(fold + 1)],
            )
            feat_imp["auc_" + str(fold + 1)] = auc
            feature_imp = pd.concat([feature_imp, feat_imp], axis=1)

            time_fold_end = diff(time_fold_start, datetime.datetime.now())
            print(
                "[Fold {}] AUC: {:2.5f} - F1-score: {:2.5f} - L. Loss: {:2.5f}  - {}".format(
                    fold + 1, auc, f1, ll, time_fold_end
                )
            )

            model_list.append(
                {"scaler": scaler_, "fold": fold + 1, "model": model_fold}
            )

        score_mean = float(np.mean(score))
        score_std = float(np.std(score))

        if score_mean > acc_best:
            acc_best = score_mean

        time_end = diff(time_start, datetime.datetime.now())
        print("-" * taco)
        print(
            "[Mean Fold] AUC: {:2.5f} std: {:2.5f} - {}".format(
                score_mean, score_std, time_end
            )
        )
        print("=" * taco)
        print()

        if save_submission_:
            name_file_sub = (
                model_name_
                + "_"
                + str(i + 1)
                + "_"
                + str(string_scaler).lower()[:4]
                + ".csv"
            )
            name_file_sub = os.path.join(path_, "Data/submission", name_file_sub)
            df_sub = df_submission.copy()
            df_sub[target_] = pred_test
            df_sub.to_csv(name_file_sub, index=False)

        if print_report_ and df_prob_temp_last is not None:
            y_pred_rep = df_prob_temp_last[
                df_prob_temp_last["scaler"] == str(string_scaler)
            ]["y_pred"]
            y_vl_rep = df_prob_temp_last[
                df_prob_temp_last["scaler"] == str(string_scaler)
            ]["y_val"]
            print(metrics.classification_report(y_vl_rep, y_pred_rep))

    print("-" * taco)
    print("Score (best mean AUC): {:2.5f}".format(acc_best))
    print("-" * taco)
    print()

    if df_prob_temp_last is None:
        df_prob_temp_last = pd.DataFrame()

    return (
        model_list,
        (
            df_prob_temp_last.sort_values(by=["scaler", "id"])
            if not df_prob_temp_last.empty
            else df_prob_temp_last
        ),
        feature_imp,
    )




## === cell 17
if _HAS_SHAP:
    try:
        X_valid_sc_small = pd.DataFrame(
            scaler_best.transform(X_valid.iloc[:5000]), columns=cols
        )
        explainer = shap.TreeExplainer(model_best)
        shap_values = explainer.shap_values(X_valid_sc_small)
        shap.summary_plot(shap_values, X_valid_sc_small, plot_type="bar", show=False)
        plt.savefig("img/shap_bar.png", bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_values, X_valid_sc_small, max_display=15, show=False)
        plt.savefig("img/shap_summary.png", bbox_inches="tight")
        plt.close()

        print("Saved SHAP plots to img/")
    except Exception as e:
        print("SHAP skipped due to:", repr(e))




## === cell 18
X_valid_sc = pd.DataFrame(scaler_best.transform(X_valid), columns=cols)
y_pred_prob = model_best.predict_proba(X_valid_sc)[:, 1]
print(
    "Final holdout AUC (for sanity): {:2.5f}".format(
        metrics.roc_auc_score(y_valid, y_pred_prob)
    )
)
print("Submission file ready at:", sub_path)
print("Submission columns:", list(pd.read_csv(sub_path, nrows=1).columns))
print("Submission rows:", sum(1 for _ in open(sub_path)) - 1)
